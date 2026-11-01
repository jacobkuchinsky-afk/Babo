"""DeepRacer SLAM bring-up: rplidar + IMU + scan filter + RF2O + EKF + slam_toolbox.

Odometry pipeline (all nodes in this one launch → reliable DDS discovery):
  rplidar          → /scan
  imu_node         → /imu  (gyro yaw-rate, accel)
  scan_filter      → /scan_filtered  (gates + full motion-comp dewarp)
  lidar_odom       → /cmd_vel  (LiDAR-measured forward velocity + vy=0)
  ekf (robot_loc)  → /odometry/filtered + odom→base_link TF  (SOLE TF owner)
  slam_toolbox     → /map + map→odom TF  (uses /scan_filtered + odom prior)
  (rf2o + cmd_vel_bridge removed — see comments below / in ekf.yaml)

Mode via SLAM_MODE env: mapping (default) | localization | lifelong.
SLAM_MAP_FILE: saved-map basename for localization.
SLAM_USE_FILTER (default '1'): '0' → slam_toolbox subscribes /scan directly and
  RF2O uses /scan (no dewarp). Normally keep on.
"""
import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


MAPPER_PARAMS = {
    'mapping':      '/home/deepracer/slam/mapper_params_mapping.yaml',
    'localization': '/home/deepracer/slam/mapper_params_localization.yaml',
    'lifelong':     '/home/deepracer/slam/mapper_params_lifelong.yaml',
}
SLAM_EXECUTABLES = {
    'mapping':      'async_slam_toolbox_node',
    'localization': 'localization_slam_toolbox_node',
    'lifelong':     'lifelong_slam_toolbox_node',
}
EKF_CONFIG = '/home/deepracer/slam/ekf.yaml'


def generate_launch_description():
    mode = os.environ.get('SLAM_MODE', 'mapping').lower()
    map_file = os.environ.get('SLAM_MAP_FILE', '').strip()
    use_filter = os.environ.get('SLAM_USE_FILTER', '1') != '0'

    if mode not in MAPPER_PARAMS:
        raise ValueError(f'invalid SLAM_MODE: {mode!r}')
    params_file = MAPPER_PARAMS[mode]
    slam_executable = SLAM_EXECUTABLES[mode]
    scan_for_slam = '/scan_filtered' if use_filter else '/scan'

    rplidar = Node(
        package='rplidar_ros',
        node_executable='rplidar_composition',
        node_name='rplidar_composition',
        output='screen',
        parameters=[{
            'serial_port': '/dev/ttyUSB0',
            'serial_baudrate': 115200,
            'frame_id': 'laser',
            'inverted': False,
            'angle_compensate': True,
        }],
    )

    imu_node = ExecuteProcess(
        cmd=['python3', '/home/deepracer/imu_node.py'],
        output='log',
    )

    # base_link -> laser static TF. THE YAW IS LOAD-BEARING (fixed June 2026):
    # the A1 is physically mounted ~92° rotated (bot nose = laser-frame +92°,
    # the push-test-calibrated NOSE_DEG used across the stack), so the laser
    # frame sits at -92° yaw in base_link. This used to be published as 0,
    # which made slam_toolbox's odometry prior point ~92° away from the true
    # motion on EVERY translation: karto commits MatchScan's best correlation
    # unconditionally (verified in Mapper.cpp), so whenever scan geometry got
    # weak/self-similar (180/360 turns, reversing, re-driving mapped areas)
    # the mis-centered search window let a wrong peak win, the bad pose
    # poisoned the running buffer, and the map cascaded into gibberish.
    # Straightaways only survived because their sharp correlation peaks beat
    # the bad prior every step. With -92° the prior is correct in all
    # maneuvers. (lidar_odom + scan_filter use the physical NOSE_DEG constant
    # and are TF-independent; slam_module converts laser<->base yaw at its
    # /initialpose + TF-seed boundaries.)
    base_to_laser = Node(
        package='tf2_ros',
        node_executable='static_transform_publisher',
        arguments=['0', '0', '0.15', '-1.6057', '0', '0', 'base_link', 'laser'],
        output='log',
    )

    actions = [rplidar, imu_node, base_to_laser]

    if use_filter:
        scan_filter = ExecuteProcess(
            cmd=['python3', '/home/deepracer/scan_filter_node.py'],
            output='log',
        )
        actions.append(scan_filter)

    # RF2O is NO LONGER LAUNCHED (2026-06). The identity base_link→laser TF
    # vs the ~92°-rotated physical mount means RF2O's twist was laser-frame:
    # its "vy" was really the bot's forward speed, which the EKF fused as a
    # lateral-zero constraint — actively wrong. It was also proven unreliable
    # on this A1 (ignored a real 2 ft push). Nothing consumes /odom_rf2o now;
    # the EKF gets vx + vy=0 from lidar_odom and vyaw from the IMU.

    # Forward-velocity source for the EKF: MEASURED from the LiDAR by scan-matching
    # (lidar_odom.py), not the open-loop throttle guess. Publishes /cmd_vel
    # (the same topic the EKF's twist0 reads), so no ekf.yaml change is needed.
    # The old throttle bridge (cmd_vel_bridge.py) is intentionally NOT launched —
    # throttle was wrong whenever the wheels didn't match the command (turns,
    # scrub, slip, coast), which froze/dragged translation and stacked the map.
    lidar_odom = ExecuteProcess(
        cmd=['python3', '/home/deepracer/lidar_odom.py'],
        output='log',
    )
    actions.append(lidar_odom)

    # EKF — fuses RF2O + IMU + cmd_vel, publishes /odometry/filtered and the
    # odom→base_link TF that slam_toolbox uses as its motion prior.
    ekf = Node(
        package='robot_localization',
        node_executable='ekf_node',
        node_name='ekf_filter_node',
        output='log',
        parameters=[EKF_CONFIG, {'use_sim_time': False}],
    )
    actions.append(ekf)

    slam_params = [params_file, {'use_sim_time': False}]
    if mode == 'localization':
        if not map_file:
            raise ValueError('SLAM_MODE=localization requires SLAM_MAP_FILE')
        slam_params.append({'map_file_name': map_file,
                            'map_start_at_dock': False})

    slam = Node(
        package='slam_toolbox',
        node_executable=slam_executable,
        node_name='slam_toolbox',
        output='screen',
        parameters=slam_params,
    )
    actions.append(slam)

    return LaunchDescription(actions)
