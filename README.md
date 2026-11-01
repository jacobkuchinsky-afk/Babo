# Babo

An AWS DeepRacer car turned into an indoor mapping robot. Babo builds a map of
the rooms it drives through with a LiDAR, works out where it is on a saved map,
and drives itself to a goal you pick, all controlled from a web dashboard
served by the car.

Everything runs on the car's Intel Atom: ROS 2 Foxy, slam_toolbox, a
robot_localization EKF, and about 8,500 lines of Python for the sensor nodes,
localization, path planning and dashboard.

<!-- TODO: dashboard screenshot or a short GIF of an autonomous run -->

## Hardware

| Part | How Babo uses it |
| --- | --- |
| AWS DeepRacer chassis | Drive and steering through the on-board PWM controller, driven directly from Python |
| RPLidar A1 (USB serial) | 360° scans for mapping, localization and odometry; mounted rotated ~92° |
| BMI160 IMU (I2C) | Yaw rate for heading, scan dewarping and steering self-calibration |
| Front camera | Live MJPEG stream in the dashboard |

## How it fits together

```
node              publishes                            read by
----------------  -----------------------------------  ------------------------------------
rplidar_ros       /scan                                scan_filter_node, lidar_odom
imu_node          /imu                                 scan_filter_node, lidar_odom, EKF
lidar_odom        /cmd_vel (measured forward speed)    EKF
EKF               odom→base_link, /odometry/filtered   scan_filter_node, slam_toolbox
scan_filter_node  /scan_filtered                       slam_toolbox
slam_toolbox      /map, map→odom                       dashboard (slam_module, auto_module)
```

The dashboard (`dashboard.py`, Flask on port 8888) starts and stops this whole
stack as one `ros2 launch` and also owns the motors, camera and e-stop.

| File | What it does |
| --- | --- |
| `src/imu_node.py` | BMI160 driver: finds the chip on the I2C bus and publishes `/imu` at 50 Hz |
| `src/scan_filter_node.py` | Drops bad LiDAR returns and undoes the motion blur in each scan (see below) |
| `src/lidar_odom.py` | Measures forward speed from the LiDAR itself by scan matching |
| `src/slam/` | Launch file, EKF config and slam_toolbox configs for mapping, localization and lifelong modes |
| `src/slam_module.py` | Starts and stops the SLAM stack, saves maps, global localization, pose confidence |
| `src/auto_module.py` | Path planning and autonomous driving |
| `src/dashboard.py` | Web dashboard and HTTP API: manual driving, camera, live map, SLAM and autonomy controls |
| `src/cmd_vel_bridge.py` | Older velocity source for the EKF, replaced by `lidar_odom.py` |

## Interesting problems

**Scans are bent when the car moves.** The A1 takes ~180 ms per sweep, and
the car turns and moves during that time, so each ray is measured from a
slightly different pose. `scan_filter_node` moves every ray back into the
pose at the start of the sweep, using the IMU for rotation and the EKF's
velocity for translation.

**A noise filter that broke mapping.** An early filter kept a point only if it
showed up in 4 of the last 5 scans. That works when standing still, but at
0.3–0.6 m/s a wall moves 5–11 cm between scans, so the filter threw away
almost every point while driving. slam_toolbox then placed scans by
prediction alone and drew the walls as a starburst. The filter was removed.

**Throttle is not speed.** The motor command is a poor guess of how fast the
car is actually going: wheels slip, scrub in turns and coast. `lidar_odom`
instead measures speed from how far the walls shift between scans. It takes
out the rotation using the IMU, finds the remaining shift with point-to-point
ICP, and feeds the result into the EKF.

**A rotated sensor that lied to the EKF.** The off-the-shelf laser odometry
(RF2O) assumed the LiDAR faced forward. Mounted ~92° off, the car's forward
speed showed up as sideways speed, and the EKF was fusing it as such. RF2O was
removed in favour of `lidar_odom`, which is calibrated to the real mounting
angle.

**Finding itself on a saved map.** slam_toolbox only searches near its current
guess, so starting somewhere new fails. `global_localize` tries every free
cell and 45 headings, projects the 720-ray scan from each, and keeps the pose
that lands the most rays on walls. Vectorized with numpy, it takes 1–3 s on
the Atom.

**Knowing when it's lost.** Pose confidence compares each pose change with the
motion that was actually commanded. Only the unexplained part counts, so
normal driving isn't flagged as "lost". When it is lost, the car stops itself.

## Autonomous driving

`auto_module` plans with 8-connected A* on the occupancy grid, inflating walls
by 0.20 m and then smoothing the path with line-of-sight checks. The car's
pose is only trusted while it's stopped, so driving is stop-and-go:

1. Stop and scan-match against the map to get an accurate pose.
2. Re-plan from there and choose forward or reverse.
3. Turn to the next heading using IMU-tracked arcs, or a three-point turn when
   space is tight.
4. Drive one short straight hop, holding heading with the IMU.

Before driving, the car checks which way its steering actually turns it with a
short test arc. It drives toward the more open side when wedged, and logs every
run to a JSONL file with 10 Hz telemetry for replaying failures.

## Running it

Requires a DeepRacer with ROS 2 Foxy, `slam_toolbox`, `robot_localization` and
`rplidar_ros` installed. Paths are hard-coded to the car's `deepracer` user.

```bash
# on the car: copy src/ to /home/deepracer (src/slam/ goes to /home/deepracer/slam)
python3 /home/deepracer/dashboard.py
```

Then open `http://<car-ip>:8888`. Mapping, localization, saved maps and
autonomous goals are all started from the dashboard. Set `SLAM_MODE` to
`mapping`, `localization` or `lifelong` when launching the SLAM stack by hand.
