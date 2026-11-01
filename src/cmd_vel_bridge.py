#!/usr/bin/env python3
"""Bridge the dashboard's command-velocity into the EKF.

The throttle/speed state lives in the dashboard process (the Driver). Pushing
it into the launch-side EKF over a ROS topic from the dashboard proved
unreliable on this hardware (cross-process /cmd_vel discovery silently failed).
HTTP from the dashboard, however, has always worked. So this node — which runs
INSIDE the SLAM launch alongside the EKF — polls the dashboard's GET /cmd_vx
endpoint and republishes it as a proper ROS twist the EKF can fuse.

Publishes: /cmd_vel  (geometry_msgs/TwistWithCovarianceStamped)
  linear.x = commanded forward velocity (m/s) = throttle * vel_factor * motor_power
  vx is the sole forward-velocity source for the EKF — RF2O's vx is dropped
  upstream in ekf.yaml because its range-flow is degenerate down a corridor.
  cmd_vel publishes at 20 Hz so the EKF's sensor_timeout (0.2 s) never trips.
"""
import json
import urllib.request

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import TwistWithCovarianceStamped


CMD_VX_URL = 'http://localhost:8888/cmd_vx'
# 20 Hz, not 5 Hz. The EKF's sensor_timeout is 0.2 s — at 5 Hz the publish
# cycle was already 200 ms, so any HTTP jitter made /cmd_vel time out and
# the filter silently fell back to RF2O alone. Down a corridor RF2O's vx
# is degenerate (~0), so the EKF reported zero forward velocity even while
# the bot was driving; slam_toolbox's prior said the bot wasn't moving and
# planted scans on top of the previous pose. 50 ms cycle gives 4× margin.
POLL_HZ = 20.0
# cmd_vel is now the SOLE vx source for the EKF (RF2O's vx is disabled in
# ekf.yaml). Tight variance keeps the prior locked to the commanded value
# between scan-match corrections.
VX_VARIANCE = 0.005


class CmdVelBridge(Node):
    def __init__(self):
        super().__init__('cmd_vel_bridge')
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=5)
        self.pub = self.create_publisher(TwistWithCovarianceStamped, '/cmd_vel', qos)
        self.timer = self.create_timer(1.0 / POLL_HZ, self._tick)
        self._warned = False
        self.get_logger().info(f'cmd_vel_bridge polling {CMD_VX_URL} at {POLL_HZ} Hz')

    def _tick(self):
        vx = 0.0
        try:
            with urllib.request.urlopen(CMD_VX_URL, timeout=0.1) as r:
                vx = float(json.loads(r.read()).get('vx', 0.0))
        except Exception as e:
            # Dashboard not reachable / endpoint missing — publish zero so the
            # EKF still gets a (low-trust) vx and doesn't time the input out.
            if not self._warned:
                self.get_logger().warn(f'cmd_vx fetch failed ({e}); publishing 0')
                self._warned = True
        msg = TwistWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.twist.linear.x = vx
        # 6x6 covariance, row-major [x,y,z,roll,pitch,yaw]; index 0 = vx.
        msg.twist.covariance[0] = VX_VARIANCE
        # Mark the unused dims with large covariance so they're ignored.
        for i in (7, 14, 21, 28, 35):
            msg.twist.covariance[i] = 1e6
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = CmdVelBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
