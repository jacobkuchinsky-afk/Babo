#!/usr/bin/env python3
"""Filter raw RPLidar A1 scans before slam_toolbox sees them.

Subscribes:  /scan, /imu
Publishes:   /scan_filtered

Pipeline:
  1. Per-point gates  — range [0.15 m .. 6.0 m], intensity >= MIN_INTENSITY.
                        Drops self-hits, far noise, and low-confidence returns.
  2. Full motion dewarp — A scan takes ~180 ms on the A1. The bot both ROTATES
                        and TRANSLATES during the sweep, so each ray was taken
                        from a slightly different pose; the raw scan looks bent
                        and smeared. We undistort every ray back into the
                        scan-start pose: rotation from the IMU yaw-rate buffer,
                        translation from the EKF's fused velocity
                        (/odometry/filtered). Each ray's hit point is rotated by
                        +dyaw and offset by the bot's displacement, then
                        converted back to (range,angle) and re-binned. Falls
                        back to rotation-only before the EKF has produced a
                        velocity estimate (startup).

We *used to* have a temporal vote here (point must appear in 4 of last 5
scans within X cm). It was great for stationary noise but catastrophic for
a moving bot: at 0.3 m/s a wall point shifts ~5 cm between scans, and at
0.6 m/s ~11 cm. With a 7 cm tolerance the filter rejected nearly every
ray during motion, leaving slam_toolbox with sparse scans it couldn't
register. Result: each scan got placed by TF prediction alone, producing
a starburst of walls at one center. Removed.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu, LaserScan
from nav_msgs.msg import Odometry

import collections
import math
import numpy as np


MIN_RANGE = 0.15       # meters — closer than this is likely the bot/mount
MAX_RANGE = 6.0        # meters — accuracy collapses past this on the A1
MIN_INTENSITY = 12.0   # A1 reports per-point quality; below this is mostly
                       # noise / partial reflections that create map fuzz
IMU_BUFFER_LEN = 200   # ~2 s at 100 Hz — generous for a ~0.18 s scan duration
NOSE_DEG = 92.0        # bot-forward (+x base_link) direction IN THE LASER FRAME
                       # (A1 mounted ~90° rotated; same constant as lidar_odom).
                       # The EKF velocity is base-frame, the scan points are
                       # laser-frame — the translation dewarp must rotate the
                       # velocity into the laser frame or it shifts points ~92°
                       # in the wrong direction (up to ~9 cm at 0.5 m/s).


class ScanFilter(Node):
    def __init__(self):
        super().__init__('scan_filter')
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=5)
        self.pub = self.create_publisher(LaserScan, '/scan_filtered', qos)
        self.sub = self.create_subscription(LaserScan, '/scan', self._on_scan, qos)
        imu_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=20)
        self.imu_sub = self.create_subscription(Imu, '/imu', self._on_imu, imu_qos)
        # IMU yaw-rate buffer (timestamp_seconds, omega_z_rad_s)
        self.imu_buf = collections.deque(maxlen=IMU_BUFFER_LEN)
        # Latest fused body-frame velocity from the EKF, for translation dewarp.
        # vy ~ 0 (non-holonomic); used with per-ray dyaw via the mid-angle rule.
        odom_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(Odometry, '/odometry/filtered', self._on_odom, odom_qos)
        self._vx = 0.0
        self._vy = 0.0
        self.get_logger().info(
            f'scan_filter ready: range=[{MIN_RANGE},{MAX_RANGE}] '
            f'intensity>={MIN_INTENSITY} (per-point gates + full motion dewarp)')

    def _on_odom(self, msg):
        self._vx = float(msg.twist.twist.linear.x)
        self._vy = float(msg.twist.twist.linear.y)

    @staticmethod
    def _stamp_seconds(stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    def _on_imu(self, msg):
        self.imu_buf.append((self._stamp_seconds(msg.header.stamp),
                             float(msg.angular_velocity.z)))

    @staticmethod
    def _build_yaw_integral(imu_buf):
        # (times, cum): cum[k] = integral of yaw-rate from times[0]..times[k]
        # with a zero-order hold (left omega held over each gap), matching the
        # old per-ray _yaw_delta accumulation. Built once per scan, O(buffer).
        if not imu_buf:
            return None, None
        m = len(imu_buf)
        arr = np.fromiter((t for t, _ in imu_buf), dtype=np.float64, count=m)
        omg = np.fromiter((w for _, w in imu_buf), dtype=np.float64, count=m)
        if m < 2:
            return arr, np.zeros(m, dtype=np.float64)
        seg = omg[:-1] * np.diff(arr)
        cum = np.concatenate(([0.0], np.cumsum(seg)))
        return arr, cum

    @staticmethod
    def _yaw_at(times, cum, t):
        # Integrated yaw from times[0] to t (scalar or array), ZOH, vectorized.
        idx = np.searchsorted(times, t, side='right') - 1
        idx = np.clip(idx, 0, len(times) - 1)
        nxt = np.minimum(idx + 1, len(times) - 1)
        gap = times[nxt] - times[idx]
        safe = np.where(gap > 0, gap, 1.0)
        omega_idx = np.where(gap > 0, (cum[nxt] - cum[idx]) / safe, 0.0)
        return cum[idx] + omega_idx * (t - times[idx])

    def _on_scan(self, msg):
        n = len(msg.ranges)
        ranges = np.asarray(msg.ranges, dtype=np.float64)
        valid = np.isfinite(ranges) & (ranges > MIN_RANGE) & (ranges < MAX_RANGE)
        if len(msg.intensities) == n:
            valid &= np.asarray(msg.intensities, dtype=np.float64) >= MIN_INTENSITY

        out = np.full(n, float('inf'), dtype=np.float64)
        times, cum = self._build_yaw_integral(self.imu_buf)
        if times is None or n == 0 or not valid.any():
            out[valid] = ranges[valid]
            self._publish(msg, out.tolist())
            return

        # Full motion dewarp (rotation from IMU yaw-rate + translation from EKF
        # velocity), vectorized: undistort each gated ray into the scan-start
        # pose. Same math as before, O(rays) numpy instead of O(rays x buffer).
        idxs = np.nonzero(valid)[0]
        r = ranges[idxs]
        scan_t0 = self._stamp_seconds(msg.header.stamp)
        scan_dur = msg.scan_time if msg.scan_time > 0 else 0.18
        ainc = msg.angle_increment
        amin = msg.angle_min
        # Rotate the base-frame EKF velocity into the laser frame (the frame
        # the points and the per-ray dyaw live in): v_laser = R(NOSE) @ v_base.
        ca, sa = math.cos(math.radians(NOSE_DEG)), math.sin(math.radians(NOSE_DEG))
        vx = ca * self._vx - sa * self._vy
        vy = sa * self._vx + ca * self._vy

        ray_t = scan_t0 + (idxs / n) * scan_dur
        elapsed = ray_t - scan_t0
        dyaw = self._yaw_at(times, cum, ray_t) - self._yaw_at(times, cum, scan_t0)

        mid = 0.5 * dyaw
        dx = (vx * np.cos(mid) - vy * np.sin(mid)) * elapsed
        dy = (vx * np.sin(mid) + vy * np.cos(mid)) * elapsed

        a_i = amin + idxs * ainc
        px_i = r * np.cos(a_i)
        py_i = r * np.sin(a_i)
        cd = np.cos(dyaw); sd = np.sin(dyaw)
        px = cd * px_i - sd * py_i + dx
        py = sd * px_i + cd * py_i + dy

        rng2 = np.hypot(px, py)
        ang2 = np.arctan2(py, px)
        if ainc != 0:
            new_i = np.rint((ang2 - amin) / ainc).astype(np.int64)
        else:
            new_i = idxs.astype(np.int64)

        keep = (new_i >= 0) & (new_i < n)
        # Nearest hit wins when multiple rays land in one bin.
        np.minimum.at(out, new_i[keep], rng2[keep])
        self._publish(msg, out.tolist())

    def _publish(self, src, ranges):
        out = LaserScan()
        out.header = src.header
        out.angle_min = src.angle_min
        out.angle_max = src.angle_max
        out.angle_increment = src.angle_increment
        out.time_increment = src.time_increment
        out.scan_time = src.scan_time
        out.range_min = src.range_min
        out.range_max = src.range_max
        out.ranges = ranges
        out.intensities = src.intensities
        self.pub.publish(out)


def main():
    rclpy.init()
    node = ScanFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
