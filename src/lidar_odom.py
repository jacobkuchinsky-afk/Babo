#!/usr/bin/env python3
"""LiDAR forward-velocity odometry for the DeepRacer.

Measures the bot's forward speed directly from the LiDAR by tracking how far
the scan points move between consecutive scans ("range flow"/scan-matching),
with the rotation removed using the IMU yaw rate. This REPLACES the open-loop
throttle estimate (cmd_vel_bridge): throttle is a guess (it's wrong whenever the
wheels don't match the command -- eased mid-turn, scrubbing, slipping, coasting,
ramps), whereas the walls are fixed in the world, so the dot shift IS the motion.

Math, per scan pair (prev -> cur):
  1. points P, C  = scan polar -> Cartesian in the laser frame.
  2. dyaw         = integral of IMU angular_velocity.z between the two scan stamps.
  3. C'           = rotate C by -dyaw   (now P and C' differ by translation only).
  4. t=(tx,ty)    = translation that best aligns C' onto P  (point-to-point ICP).
                    The bot's displacement in the laser frame equals +t.
  5. forward speed= component of t along the measured nose axis (NOSE_DEG in the
                    laser frame), divided by dt. The nose axis was calibrated by
                    pushing the bot and reading the motion direction: ~+92 deg
                    (the A1 is mounted rotated ~90 deg on this car). Magnitude is
                    rotation-invariant, so a few degrees of error is negligible.

Publishes: /cmd_vel (geometry_msgs/TwistWithCovarianceStamped), linear.x = forward
m/s, base_link frame, at PUBLISH_HZ (held between scans so the EKF's 0.2 s
sensor_timeout never trips). Velocity is zeroed if scans stop arriving.

Low-speed note: a few cm/s of motion is < the A1's range noise, so this reports
~0 when creeping. That's fine -- at mapping speed the dot shift is large, and
near-stationary motion barely changes the map anyway. vyaw still comes from the
IMU and vy=0 from the non-holonomic constraint (RF2O), as before.
"""
import os, math, time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import TwistWithCovarianceStamped

SCAN_TOPIC = '/scan'          # raw scan; range-flow wants un-dewarped geometry
NOSE_DEG = 92.0               # forward (+x base_link) direction in the LASER frame,
                              # calibrated by push test. Forward speed = t . nose_hat.
MAX_PTS = 320                 # subsample cap (Atom CPU budget)
ICP_ITERS = 6
MIN_PTS = 40                  # need this many points to trust a match
MIN_CORR = 25                 # min inlier correspondences
MAX_SPEED = 3.0               # m/s sanity clamp; faster than this = bad match -> reject
VMIN = 0.03                   # m/s dead-band: below this is ICP noise, not motion.
                              # Zeroes it so a parked bot doesn't slowly drift (a
                              # few cm/s of noise integrates into ghost translation).
                              # Real mapping speed is >> this, so nothing is lost.
PUBLISH_HZ = 20.0             # keep EKF vx fed (sensor_timeout 0.2 s)
SCAN_STALE_S = 0.4            # no scan for this long -> publish 0
VX_VARIANCE = 0.03            # measured velocity; honest (vs throttle guess 0.005)
VY_VARIANCE = 0.04            # soft non-holonomic vy=0 constraint for the EKF
EMA = 0.6                     # light smoothing on the per-scan speed


def _sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def _to_xy(msg):
    r = np.asarray(msg.ranges, dtype=np.float64)
    th = msg.angle_min + np.arange(len(r)) * msg.angle_increment
    ok = np.isfinite(r) & (r > 0.15) & (r < 6.0)
    p = np.stack([r[ok] * np.cos(th[ok]), r[ok] * np.sin(th[ok])], 1)
    if len(p) > MAX_PTS:
        p = p[np.linspace(0, len(p) - 1, MAX_PTS).astype(int)]
    return p


def _icp_translation(cur, prev):
    """Translation-only ICP (rotation already removed). Returns (tx,ty,inliers)."""
    tx = ty = 0.0
    inl = 0
    for _ in range(ICP_ITERS):
        q = cur + np.array([tx, ty])
        d2 = ((q[:, None, :] - prev[None, :, :]) ** 2).sum(2)
        nn = d2.argmin(1)
        diff = prev[nn] - q
        dd = np.sqrt((diff ** 2).sum(1))
        gate = np.median(dd) * 2.0 + 0.05
        msk = dd < gate
        inl = int(msk.sum())
        if inl < MIN_CORR:
            break
        step = diff[msk].mean(0)
        tx += step[0]; ty += step[1]
        if abs(step[0]) + abs(step[1]) < 1e-4:
            break
    return tx, ty, inl


class LidarOdom(Node):
    def __init__(self):
        super().__init__('lidar_odom')
        qs = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST, depth=5)
        qi = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                        history=HistoryPolicy.KEEP_LAST, depth=20)
        self.create_subscription(LaserScan, SCAN_TOPIC, self.on_scan, qs)
        self.create_subscription(Imu, '/imu', self.on_imu, qi)
        self.pub = self.create_publisher(TwistWithCovarianceStamped, '/cmd_vel', qs)

        a = math.radians(NOSE_DEG)
        self.nose = np.array([math.cos(a), math.sin(a)])   # forward unit, laser frame

        self.yaw = 0.0
        self.last_imu_t = None
        self.prev_pts = None
        self.prev_yaw = None
        self.prev_t = None
        self.vx = 0.0
        self.last_scan_t = 0.0

        self.create_timer(1.0 / PUBLISH_HZ, self.publish)
        self.get_logger().info(
            f'lidar_odom: scan-matching forward velocity on {SCAN_TOPIC}, '
            f'nose={NOSE_DEG} deg, publishing /cmd_vel at {PUBLISH_HZ} Hz')

    def on_imu(self, m):
        t = _sec(m.header.stamp)
        if self.last_imu_t is not None:
            dt = t - self.last_imu_t
            if 0.0 < dt < 0.1:
                self.yaw += m.angular_velocity.z * dt
        self.last_imu_t = t

    def on_scan(self, m):
        t = _sec(m.header.stamp)
        pts = _to_xy(m)
        yaw = self.yaw
        if (self.prev_pts is not None and len(pts) >= MIN_PTS
                and len(self.prev_pts) >= MIN_PTS):
            dyaw = yaw - self.prev_yaw
            dt = t - self.prev_t
            if dt > 0:
                # Map the new scan back into the PREVIOUS laser frame: the
                # bot rotated +dyaw (CCW), so the new frame's coords need a
                # +dyaw rotation to land in the old frame (same direction as
                # scan_filter's dewarp: p_old = R(+dyaw) @ p_new). This was
                # R(-dyaw) before, which DOUBLED the rotation instead of
                # removing it and corrupted measured speed during turns.
                c, s = math.cos(dyaw), math.sin(dyaw)
                cur = pts @ np.array([[c, s], [-s, c]])      # row-vec R(+dyaw)
                tx, ty, inl = _icp_translation(cur, self.prev_pts)
                if inl >= MIN_CORR:
                    fwd = float(np.dot([tx, ty], self.nose)) / dt   # forward m/s
                    if abs(fwd) <= MAX_SPEED:
                        if abs(fwd) < VMIN:      # dead-band: treat as stationary
                            fwd = 0.0
                        self.vx = EMA * fwd + (1.0 - EMA) * self.vx
                        self.last_scan_t = time.monotonic()
        self.prev_pts = pts
        self.prev_yaw = yaw
        self.prev_t = t

    def publish(self):
        fresh = (time.monotonic() - self.last_scan_t) < SCAN_STALE_S
        vx = self.vx if fresh else 0.0
        if not fresh:
            self.vx = 0.0
        msg = TwistWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.twist.linear.x = vx
        # linear.y stays 0.0: the car is non-holonomic, so this twist is also
        # the EKF's lateral-zero constraint (used to come from RF2O, but RF2O's
        # "vy" was really laser-frame ≈ forward speed thanks to the 92° mount).
        msg.twist.covariance[0] = VX_VARIANCE if fresh else 1.0
        msg.twist.covariance[7] = VY_VARIANCE
        for i in (14, 21, 28, 35):
            msg.twist.covariance[i] = 1e6
        self.pub.publish(msg)


def main():
    rclpy.init()
    n = LidarOdom()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
