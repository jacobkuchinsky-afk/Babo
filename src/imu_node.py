#!/usr/bin/env python3
"""BMI160 IMU sensor publisher.

Reads the on-board BMI160 over /dev/i2c-{1 or 2} (auto-detected) at 50 Hz
and publishes a single topic:

  /imu   sensor_msgs/Imu       — full IMU data (orientation, gyro, accel)

Previously this node also published /odom and the odom->base_link TF, but
cross-process /cmd_vel discovery was unreliable on this hardware so the
node never received the commanded velocity needed for translation
integration. The dashboard process now owns odometry: it subscribes to
/imu, reads driver state in-process, and publishes TF itself.

This node is now a thin sensor driver. CHIP_ID 0xD1 confirms BMI160."""
import fcntl
import math
import struct
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu


I2C_BUS = '/dev/i2c-1'   # default — overridden by auto-probe at startup
I2C_ADDR = 0x68
I2C_SLAVE = 0x0703  # from linux/i2c-dev.h


def _find_bmi160_bus():
    """Scan /dev/i2c-* for a device at 0x68 that returns BMI160 CHIP_ID 0xD1.

    The i2c bus numbering on this hardware isn't stable across reboots (it
    re-enumerates depending on driver load order). Hardcoding /dev/i2c-1
    worked once and silently broke on the next boot. Probe each bus instead."""
    import glob
    for path in sorted(glob.glob('/dev/i2c-*')):
        try:
            fd = open(path, 'r+b', buffering=0)
            fcntl.ioctl(fd, I2C_SLAVE, I2C_ADDR)
            fd.write(bytes([0x00]))   # CHIP_ID register
            b = fd.read(1)
            fd.close()
            if b and b[0] == 0xD1:
                return path
        except OSError:
            try: fd.close()
            except Exception: pass
    return None

# BMI160 registers
CHIP_ID_REG = 0x00
ERR_REG     = 0x01
PMU_STATUS  = 0x02
DATA_GYR    = 0x0C    # 6 bytes LE: GYR x/y/z
DATA_ACC    = 0x12    # 6 bytes LE: ACC x/y/z
ACC_CONF    = 0x40
ACC_RANGE   = 0x41
GYR_CONF    = 0x42
GYR_RANGE   = 0x43
CMD_REG     = 0x7E

# CMD values (BMI160 datasheet §3.3.4)
CMD_ACC_NORMAL = 0x11    # accel -> normal mode, takes ~4 ms
CMD_GYR_NORMAL = 0x15    # gyro  -> normal mode, takes ~80 ms
CMD_SOFTRESET  = 0xB6

# Scaling for the ranges we configure (±250 °/s, ±2 g)
GYRO_SCALE = 131.2       # LSB per °/s at ±250
ACCEL_SCALE = 16384.0    # LSB per g  at ±2
G = 9.80665              # m/s² per g

# The BMI160 is mounted z-down on this car, so its gyro z reads CW-positive
# while ROS/REP-103 (and RF2O, the EKF, slam_toolbox, the scan dewarp) all
# expect CCW-positive yaw. Without this the estimated heading turned the wrong
# way on every turn: slam_toolbox's motion prior pointed opposite the real
# rotation, so scans were planted at the wrong angle (misplaced/doubled walls),
# and the IMU fought RF2O inside the EKF. Verified by hand-rotation: a physical
# +90° CCW turn integrated to -94° before this flip. Negating at the source
# corrects every downstream consumer. See yaw_run.py for the diagnostic.
GYRO_Z_SIGN = -1.0

SAMPLE_HZ = 50
BIAS_SAMPLES = 200


def _yaw_to_quat(yaw_rad):
    return (0.0, 0.0, math.sin(yaw_rad / 2), math.cos(yaw_rad / 2))


class BMI160:
    def __init__(self, bus_path=None, addr=I2C_ADDR):
        if bus_path is None:
            bus_path = _find_bmi160_bus()
            if bus_path is None:
                raise RuntimeError(
                    'BMI160 (CHIP_ID 0xD1) not found on any /dev/i2c-* bus')
        self.bus_path = bus_path
        self.fd = open(bus_path, 'r+b', buffering=0)
        fcntl.ioctl(self.fd, I2C_SLAVE, addr)

    def _read(self, reg, n):
        self.fd.write(bytes([reg]))
        return self.fd.read(n)

    def _write(self, reg, val):
        self.fd.write(bytes([reg, val]))

    def init(self):
        # Soft reset, then power up accel + gyro, then set ranges.
        self._write(CMD_REG, CMD_SOFTRESET)
        time.sleep(0.05)
        chip_id = self._read(CHIP_ID_REG, 1)[0]
        if chip_id != 0xD1:
            raise RuntimeError(f'unexpected BMI160 CHIP_ID 0x{chip_id:02x} (expected 0xD1)')
        self._write(CMD_REG, CMD_ACC_NORMAL); time.sleep(0.005)
        self._write(CMD_REG, CMD_GYR_NORMAL); time.sleep(0.085)
        # ACC_CONF: 0x28 = no oversample, ODR 100 Hz (low-pass filter normal)
        self._write(ACC_CONF, 0x28)
        # ACC_RANGE: 0x03 = ±2 g
        self._write(ACC_RANGE, 0x03)
        # GYR_CONF: 0x28 = ODR 100 Hz, BWP normal
        self._write(GYR_CONF, 0x28)
        # GYR_RANGE: 0x03 = ±250 °/s
        self._write(GYR_RANGE, 0x03)
        time.sleep(0.05)
        return chip_id

    def read_raw(self):
        # 12 bytes: gyro x/y/z then accel x/y/z, little-endian signed shorts.
        b = self._read(DATA_GYR, 12)
        gx, gy, gz, ax, ay, az = struct.unpack('<hhhhhh', b)
        return ax, ay, az, gx, gy, gz


class ImuNode(Node):
    def __init__(self):
        super().__init__('imu_node')
        self.imu = BMI160()
        chip_id = self.imu.init()
        self.get_logger().info(
            f'BMI160 CHIP_ID=0x{chip_id:02x} on {self.imu.bus_path} initialized')

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
        self.imu_pub = self.create_publisher(Imu, '/imu', qos)

        self.get_logger().info(
            f'calibrating gyro bias ({BIAS_SAMPLES} samples, ~{BIAS_SAMPLES/100:.1f}s — keep still)')
        gx_sum = gy_sum = gz_sum = 0.0
        for _ in range(BIAS_SAMPLES):
            _, _, _, gx, gy, gz = self.imu.read_raw()
            gx_sum += gx; gy_sum += gy; gz_sum += gz
            time.sleep(0.01)
        self.bias_gx = gx_sum / BIAS_SAMPLES
        self.bias_gy = gy_sum / BIAS_SAMPLES
        self.bias_gz = gz_sum / BIAS_SAMPLES
        self.get_logger().info(
            f'gyro bias (raw): x={self.bias_gx:.1f} y={self.bias_gy:.1f} z={self.bias_gz:.1f}')

        self.yaw = 0.0   # integrated for the orientation field on /imu
        self.last_time = self.get_clock().now()

        self.timer = self.create_timer(1.0 / SAMPLE_HZ, self._tick)

    def _tick(self):
        try:
            ax, ay, az, gx, gy, gz = self.imu.read_raw()
        except Exception as e:
            self.get_logger().warn(f'imu read failed: {e}', throttle_duration_sec=2)
            return

        gx_dps = (gx - self.bias_gx) / GYRO_SCALE
        gy_dps = (gy - self.bias_gy) / GYRO_SCALE
        # GYRO_Z_SIGN flips the z-down mounting into REP-103 CCW-positive yaw.
        gz_dps = (gz - self.bias_gz) / GYRO_SCALE * GYRO_Z_SIGN
        ax_ms2 = (ax / ACCEL_SCALE) * G
        ay_ms2 = (ay / ACCEL_SCALE) * G
        az_ms2 = (az / ACCEL_SCALE) * G

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        if dt > 0.1:
            dt = 1.0 / SAMPLE_HZ
        gz_rad = gz_dps * math.pi / 180.0
        self.yaw += gz_rad * dt
        if self.yaw > math.pi:
            self.yaw -= 2 * math.pi
        elif self.yaw < -math.pi:
            self.yaw += 2 * math.pi

        qx, qy, qz, qw = _yaw_to_quat(self.yaw)
        stamp = now.to_msg()

        imu_msg = Imu()
        imu_msg.header.stamp = stamp
        imu_msg.header.frame_id = 'base_link'
        imu_msg.orientation.x = qx
        imu_msg.orientation.y = qy
        imu_msg.orientation.z = qz
        imu_msg.orientation.w = qw
        imu_msg.orientation_covariance = [0.05, 0., 0.,  0., 0.05, 0.,  0., 0., 0.05]
        imu_msg.angular_velocity.x = gx_dps * math.pi / 180.0
        imu_msg.angular_velocity.y = gy_dps * math.pi / 180.0
        imu_msg.angular_velocity.z = gz_rad
        imu_msg.angular_velocity_covariance = [0.01, 0., 0.,  0., 0.01, 0.,  0., 0., 0.01]
        imu_msg.linear_acceleration.x = ax_ms2
        imu_msg.linear_acceleration.y = ay_ms2
        imu_msg.linear_acceleration.z = az_ms2
        imu_msg.linear_acceleration_covariance = [0.5, 0., 0.,  0., 0.5, 0.,  0., 0., 0.5]
        self.imu_pub.publish(imu_msg)

        # /odom + TF odom→base_link are now published by the dashboard process
        # using this /imu's yaw rate + in-process driver state. See module
        # docstring for why we moved that responsibility.


def main():
    rclpy.init()
    node = ImuNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
