"""SLAM lifecycle manager for the dashboard.

Owns:
  - the `ros2 launch` subprocess that runs rplidar + static TFs + slam_toolbox
  - a background rclpy node that subscribes to /scan, /map, and TF map->base_link
  - serial-port handover with the dashboard's existing Lidar class (only one
    process can own /dev/ttyUSB0 at a time)
"""
import glob
import math
import os
import signal
import subprocess
import sys
import threading
import time


LAUNCH_FILE = '/home/deepracer/slam/slam_launch.py'
MAP_DIR = '/home/deepracer/slam/maps'
ROS_SETUP = '/opt/ros/foxy/setup.bash'
RF2O_WS_SETUP = '/home/deepracer/ros2_ws/install/setup.bash'  # for rf2o_laser_odometry
SCAN_TOPIC = '/scan'
MAP_TOPIC = '/map'
CMD_VEL_TOPIC = '/cmd_vel'

VALID_MODES = ('mapping', 'localization', 'lifelong')

# Persistent yaw-offset (degrees) applied to every global_localize result
# before publishing /initialpose. Lets the user manually correct a stuck
# yaw offset (e.g. when the bot's reported orientation is always 90° off
# physical reality). Range: [-180, 180]. Stored in this file across runs.
YAW_OFFSET_FILE = '/home/deepracer/.slam_yaw_offset_deg'

# Bot-forward direction in the laser frame (A1 mounted ~92° rotated; the
# push-test-calibrated NOSE_DEG). Two distinct uses:
#  * _predict_motion: forward travel in map frame = laser_yaw + NOSE.
#  * Boundary conversion with slam_toolbox: since the base_link->laser static
#    TF now tells the truth (yaw = -92°, see slam_launch.py), slam_toolbox's
#    BASE yaw differs from this module's internal LASER yaw by +92°:
#    theta_base = theta_laser + LASER_NOSE_RAD. The custom matcher works in
#    raw laser rays, so everything in this module (self._pose, the dashboard
#    overlay/arrow, AutoNav's theta+92 nose math, saved-map matching) stays
#    laser-yaw; we convert ONLY when publishing /initialpose (laser->base)
#    and when seeding the pose from the map->base_link TF (base->laser).
#    Old saved maps keep working: karto matches SENSOR poses, and the
#    raw-ray sensor convention never changed.
LASER_NOSE_RAD = math.radians(92.0)


def _load_yaw_offset_deg():
    try:
        with open(YAW_OFFSET_FILE) as f:
            v = float(f.read().strip())
        if -180.0 <= v <= 180.0:
            return v
    except (OSError, ValueError):
        pass
    return 0.0


def _save_yaw_offset_deg(v):
    try:
        with open(YAW_OFFSET_FILE, 'w') as f:
            f.write(f'{v}\n')
    except OSError as e:
        _log(f'yaw offset save failed: {e}')


def _log(msg):
    # Timestamped so dashboard.log merges with the auto-drive JSONL run logs.
    ts = time.time()
    stamp = time.strftime('%H:%M:%S', time.localtime(ts)) + f'.{int(ts % 1 * 1000):03d}'
    print(f'[slam {stamp}] {msg}', file=sys.stderr, flush=True)


class SlamManager:
    """Lifecycle: idle -> starting -> active -> stopping -> idle.

    Transitions happen on background threads; HTTP handlers see consistent state
    via the lock and a snapshot of cached telemetry."""

    def __init__(self, serial_lidar=None, driver=None):
        self._serial_lidar = serial_lidar
        # Driver reference is used to publish /cmd_vel while SLAM is active —
        # gives slam_toolbox a motion prior derived from what we're commanding
        # the motors to do. Without this, scan-matching has to invent motion
        # from scratch, which falls apart on noisy A1 scans.
        self._driver = driver
        self._lock = threading.Lock()
        self.state = 'idle'
        self.error = None

        # Last-used start parameters (so /slam/reset can reuse them).
        self.mode = 'mapping'
        self.map_file = ''   # basename in MAP_DIR (no extension)

        self._launch_proc = None
        self._ros_stop = threading.Event()
        self._ros_thread = None

        self._scan_points = []
        self._scan_ts = 0.0
        self._scan_rate_hz = 0.0
        self._scan_count = 0
        self._scan_window_start = 0.0

        self._map = None  # nav_msgs.msg.OccupancyGrid
        self._map_ts = 0.0
        self._map_known_cells = 0
        self._map_occupied_cells = 0

        self._pose = None  # (x, y, theta) in map frame
        self._pose_ts = 0.0

        # Diagnostic plumbing — track what we're commanding and what imu_node
        # is integrating, so a UI/HTTP client can see the full odometry chain
        # without trying to subscribe from a separate process (DDS discovery
        # across processes on this hardware is flaky).
        self._last_cmd_vx = 0.0       # most recent linear.x we published to /cmd_vel
        self._odom_xy = None          # (x, y) from /odom — the imu_node integration
        self._odom_yaw = 0.0          # yaw (radians) from the EKF odom→base_link TF
        self._odom_ts = 0.0
        # Pose-confidence heuristic. Tracks recent pose jumps; if SLAM is
        # losing track, pose will start jumping in big steps between updates.
        self._pose_history = []   # list of (x, y, theta, ts, cmd_vx)
        self._confidence = 1.0    # 0..1 — set by _update_confidence()
        self._lost_callback = None  # called when confidence drops to "lost"
        self._was_lost = False    # so we don't spam the callback

        # Active-localization "jiggle": gently rotates and nudges to disambiguate
        # initial localization. The first scan-match success often produces a
        # large pose snap from (0,0,0) to the true location, which the lost-
        # confidence heuristic correctly flags as anomalous — during a jiggle
        # we suppress the lost callback so localization can succeed without
        # tripping the safety auto-stop. Restored on jiggle exit.
        self._jiggle_thread = None
        self._jiggle_stop = threading.Event()
        self._jiggle_status = 'idle'   # idle | running | done | cancelled | error

        # Global-localization publisher; set up in _ros_spin once the node
        # exists. Used to seed slam_toolbox with a known-good pose by
        # publishing PoseWithCovarianceStamped on /initialpose.
        self._initialpose_pub = None
        self._global_localize_status = 'idle'  # idle | running | success | fail
        # Cache of the most recent global_localize result. Callers that
        # need the post-localize pose immediately (e.g. replan-after-align)
        # can use this instead of waiting for slam_toolbox's TF round-trip.
        self._last_globalize_pose = None
        self._last_globalize_ts = 0.0
        self._last_globalize_hit_frac = 0.0
        # User-tunable yaw-offset applied to every global_localize result
        # before publishing /initialpose. Lets the user correct a persistent
        # rotational error in localization (e.g. always-90°-off).
        self._yaw_offset_rad = math.radians(_load_yaw_offset_deg())
        # Continuous local-scan-match tracking: a background thread that
        # runs local_scan_match every second while SLAM is in localization
        # mode. Keeps the bot's pose anchored to the live scan-vs-map match
        # so the LiDAR points stay on the map walls during driving.
        # Without these fields initialised here, start_continuous_localize
        # crashes with AttributeError — which silently killed continuous
        # tracking in the previous build, letting pose drift off the map.
        self._continuous_localize_thread = None
        self._continuous_localize_stop = threading.Event()
        # Pause flag for the continuous loop. AutoNav sets this while a drive
        # runs: TWO localizers fighting over self._pose was the run-2 circling
        # bug — the 1 Hz ±25° tracker can't follow a full-lock turn (~40°/s),
        # committed corridor-symmetric garbage poses mid-maneuver, AND held
        # the match lock so AutoNav's own relocalize fell back to adopting
        # those garbage poses. While driving, AutoNav is the sole localizer.
        self._continuous_paused = False
        # Exclusivity lock: prevents global_localize and local_scan_match
        # from running concurrently (e.g. continuous loop fires while an
        # auto-drive flow also triggers a heavy global_localize).
        self._localize_in_progress = threading.Lock()
        # Pose pin: when global_localize or local_scan_match commits a
        # correction we set this timestamp to "now + pin window". Until
        # then, _ros_spin's TF lookup MUST NOT overwrite self._pose —
        # slam_toolbox can be slow to absorb /initialpose into its TF
        # tree, and without the pin our scan-match correction gets
        # immediately reverted by the next stale TF lookup.
        # 1.5 s pin + 1.0 s continuous-localize interval means the next
        # local match runs against our pinned pose (good), but TF can
        # catch up if it eventually does (also good).
        self._pose_pin_until_ts = 0.0
        self.POSE_PIN_S = 1.5
        # LiDAR-measured forward velocity from the EKF (/odometry/filtered,
        # fed by lidar_odom). Used for x/y motion prediction and for the
        # expected-motion allowance in the lost-confidence heuristic. Unlike
        # _last_cmd_vx (open-loop throttle guess) this is a real measurement.
        self._last_ekf_vx = 0.0
        self._last_ekf_vx_ts = 0.0
        # Consecutive 'no good match' failures from the continuous localize
        # loop — drives the escalating recovery (±180° yaw sweep, then
        # global_localize). Large yaw errors (e.g. 180° flips) are invisible
        # to the normal ±25° local search; this is how we climb out.
        self._match_fail_streak = 0

    def set_yaw_offset_deg(self, deg):
        """Set the yaw-offset (degrees, [-180, 180]) AND immediately apply
        the delta to the bot's current pose by republishing /initialpose
        with the rotated yaw. This way moving the slider has an instant
        visible effect — the bot icon rotates by the slider delta in real
        time — instead of only taking effect on the next global_localize.

        The new offset also persists across restarts and is applied to
        every future global_localize result."""
        deg = max(-180.0, min(180.0, float(deg)))
        new_offset_rad = math.radians(deg)
        with self._lock:
            old_offset_rad = self._yaw_offset_rad
            self._yaw_offset_rad = new_offset_rad
            current_pose = self._pose
        _save_yaw_offset_deg(deg)
        delta_rad = new_offset_rad - old_offset_rad

        # If SLAM has a current pose, rotate it by the delta and publish.
        # Apply via /initialpose so slam_toolbox absorbs the correction
        # into its map→odom TF — subsequent tracking continues from the
        # corrected yaw.
        applied = False
        if (abs(delta_rad) > 1e-4
                and current_pose is not None
                and self._initialpose_pub is not None):
            x, y, theta = current_pose
            new_yaw = theta + delta_rad
            # Wrap to [-pi, pi]
            new_yaw = (new_yaw + math.pi) % (2 * math.pi) - math.pi
            self._publish_initial_pose(x, y, new_yaw)
            # Refresh the cached pose so anyone reading it gets the new yaw
            # without waiting for the TF round-trip.
            with self._lock:
                self._pose = (x, y, new_yaw)
                # Also update the cached global_localize pose if we have one
                if self._last_globalize_pose is not None:
                    gx, gy, _ = self._last_globalize_pose
                    self._last_globalize_pose = (gx, gy, new_yaw)
            applied = True
        if applied:
            _log(f'yaw offset → {deg:+.1f}°, rotated current pose '
                 f'by {math.degrees(delta_rad):+.1f}° (live)')
        else:
            _log(f'yaw offset → {deg:+.1f}° (no live pose; will apply '
                 'on next global_localize)')
        return deg

    def get_yaw_offset_deg(self):
        with self._lock:
            return round(math.degrees(self._yaw_offset_rad), 2)

    def local_scan_match(self, search_radius_m=0.50, search_yaw_deg=25.0,
                         step_m=0.10, step_deg=3.0,
                         min_hit_fraction=0.18, prior_pose=None):
        """Fast scan-vs-map refinement around the bot's pose.

        If `prior_pose` is given, the search is centered there. Otherwise
        it centers on self._pose. This lets the continuous-localize loop
        pass a motion-predicted pose without mutating self._pose — a
        failed match then leaves self._pose at its last verified value
        rather than drifting.

        Returns (new_pose, msg) on a successful publish, (None, reason)
        otherwise."""
        try:
            import numpy as np
        except ImportError:
            return None, 'numpy unavailable'

        if not self._localize_in_progress.acquire(blocking=False):
            return None, 'global_localize busy'
        try:
            return self._local_scan_match_impl(
                np, search_radius_m, search_yaw_deg, step_m, step_deg,
                min_hit_fraction, prior_pose)
        finally:
            self._localize_in_progress.release()

    def _local_scan_match_impl(self, np, search_radius_m, search_yaw_deg,
                               step_m, step_deg, min_hit_fraction,
                               prior_pose=None):
        with self._lock:
            grid = self._map
            scan = list(self._scan_points)
            current = self._pose
        # Search prior: caller's prediction if given, else last verified.
        search_center = prior_pose if prior_pose is not None else current
        if grid is None or search_center is None or len(scan) < 30:
            return None, 'no map / scan / pose'

        cx0, cy0, theta0 = search_center
        w = grid.info.width
        h = grid.info.height
        res = grid.info.resolution
        ox = grid.info.origin.position.x
        oy = grid.info.origin.position.y
        map_arr = np.asarray(grid.data, dtype=np.int8).reshape((h, w))
        occupied = (map_arr > 50)
        free_or_unknown = ~occupied
        x_hi = ox + w * res
        y_hi = oy + h * res

        scan_xy = np.array([
            (d * math.cos(math.radians(a)) / 1000.0,
             d * math.sin(math.radians(a)) / 1000.0)
            for a, d in scan
            if 100.0 < d < 5500.0
        ], dtype=np.float32)
        if len(scan_xy) < 30:
            return None, f'only {len(scan_xy)} usable rays'
        rx = scan_xy[:, 0]
        ry = scan_xy[:, 1]
        # Apply yaw_offset by PRE-ROTATING the scan rays. Effect: the matcher
        # operates in the user-corrected frame, so its output yaw is the
        # physical yaw directly — no post-match addition needed and the
        # correction can't be undone by a subsequent local match (which
        # was the bug that reverted the user's -90° offset on every cycle).
        with self._lock:
            offset_rad = self._yaw_offset_rad
        if abs(offset_rad) > 1e-4:
            beta = -offset_rad
            cb = math.cos(beta); sb = math.sin(beta)
            new_rx = cb * rx - sb * ry
            new_ry = sb * rx + cb * ry
            rx, ry = new_rx, new_ry
        n_rays = len(rx)

        # Candidate poses in a small neighborhood of current.
        xs = np.arange(cx0 - search_radius_m,
                       cx0 + search_radius_m + step_m / 2,
                       step_m, dtype=np.float32)
        ys = np.arange(cy0 - search_radius_m,
                       cy0 + search_radius_m + step_m / 2,
                       step_m, dtype=np.float32)
        yaws = np.radians(np.arange(
            math.degrees(theta0) - search_yaw_deg,
            math.degrees(theta0) + search_yaw_deg + step_deg / 2,
            step_deg, dtype=np.float32))

        # Filter candidates to free cells only (bot can't be in a wall).
        cands = []
        for cx in xs:
            mx = int((cx - ox) / res)
            if not (0 <= mx < w):
                continue
            for cy in ys:
                my = int((cy - oy) / res)
                if not (0 <= my < h):
                    continue
                if free_or_unknown[my, mx]:
                    cands.append((cx, cy))
        if not cands:
            return None, 'no free candidates near current pose'
        cand_arr = np.array(cands, dtype=np.float32)
        cx_arr = cand_arr[:, 0]
        cy_arr = cand_arr[:, 1]

        check_fractions = (0.35, 0.65, 0.90)
        per_yaw_scores = []   # (yaw, per-candidate hit counts)
        for yaw in yaws:
            cos_y = float(np.cos(yaw))
            sin_y = float(np.sin(yaw))
            dx = cos_y * rx - sin_y * ry
            dy = sin_y * rx + cos_y * ry
            wx = cx_arr[:, None] + dx[None, :]
            wy = cy_arr[:, None] + dy[None, :]
            inb = ((wx >= ox) & (wx < x_hi) & (wy >= oy) & (wy < y_hi))
            mx = np.clip(((wx - ox) / res).astype(np.int32), 0, w - 1)
            my = np.clip(((wy - oy) / res).astype(np.int32), 0, h - 1)
            end_occ = occupied[my, mx] & inb
            inter = np.ones_like(end_occ)
            for f in check_fractions:
                fwx = cx_arr[:, None] + f * dx[None, :]
                fwy = cy_arr[:, None] + f * dy[None, :]
                fin = ((fwx >= ox) & (fwx < x_hi) &
                       (fwy >= oy) & (fwy < y_hi))
                fmx = np.clip(((fwx - ox) / res).astype(np.int32),
                              0, w - 1)
                fmy = np.clip(((fwy - oy) / res).astype(np.int32),
                              0, h - 1)
                inter &= (free_or_unknown[fmy, fmx] & fin)
            good = end_occ & inter
            per_yaw_scores.append((float(yaw), good.sum(axis=1)))

        # FLAT-WALL FIX: along a featureless wall (the aperture problem)
        # sliding the candidate parallel to the wall barely changes the hit
        # count, so many candidates tie. The old code took argmax(), and
        # numpy's argmax returns the FIRST max — i.e. the most-negative-x/y
        # candidate — a systematic corner-ward pose drift precisely on flat
        # walls. Instead, among all candidates within 5% of the best score,
        # commit to the one CLOSEST to the search prior (position plus yaw,
        # 1 rad weighted as 0.8 m): the wall constrains the perpendicular
        # axis, odometry/prediction keeps the parallel axis honest.
        best_raw = max(int(s.max()) for _, s in per_yaw_scores)
        if best_raw <= 0:
            return None, f'no good match (0/{n_rays})'
        thresh = max(1, int(math.ceil(best_raw * 0.95)))
        best_pose = None
        best_cost = float('inf')
        best_score = -1
        for yaw, scores in per_yaw_scores:
            dyaw_c = abs(((yaw - theta0) + math.pi) % (2 * math.pi) - math.pi)
            for ci in np.nonzero(scores >= thresh)[0]:
                cost = (math.hypot(float(cx_arr[ci]) - cx0,
                                   float(cy_arr[ci]) - cy0) + 0.8 * dyaw_c)
                if cost < best_cost:
                    best_cost = cost
                    best_score = int(scores[ci])
                    best_pose = (float(cx_arr[ci]), float(cy_arr[ci]), yaw)

        if best_pose is None or best_score < min_hit_fraction * n_rays:
            return None, f'no good match ({best_score}/{n_rays})'

        # Only publish if the new pose is meaningfully different — no point
        # sending /initialpose every tick if nothing changed.
        dxs = best_pose[0] - cx0
        dys = best_pose[1] - cy0
        dyaw = ((best_pose[2] - theta0) + math.pi) % (2 * math.pi) - math.pi
        moved = math.hypot(dxs, dys) > 0.05 or abs(dyaw) > math.radians(2.0)
        if not moved:
            return best_pose, f'no meaningful change ({best_score}/{n_rays})'

        # Edge-clip warning: if the best match lands at the edge of the
        # Edge-clip detection — log only, don't reject. (Rejecting these
        # was making the icon stick to stale poses while the bot drove,
        # which was visibly worse.)
        clipped_x = abs(dxs) > search_radius_m * 0.9
        clipped_y = abs(dys) > search_radius_m * 0.9
        clipped_yaw = abs(dyaw) > math.radians(search_yaw_deg * 0.9)
        edge = []
        if clipped_x: edge.append('x')
        if clipped_y: edge.append('y')
        if clipped_yaw: edge.append('yaw')

        ok = self._publish_initial_pose(*best_pose)
        if not ok:
            return None, 'publish failed'
        with self._lock:
            self._pose = best_pose
            self._pose_ts = time.monotonic()
            self._pose_pin_until_ts = time.monotonic() + self.POSE_PIN_S
        edge_msg = f' EDGE-CLIP[{",".join(edge)}]' if edge else ''
        return best_pose, (f'local match: ({best_pose[0]:.2f}, '
                           f'{best_pose[1]:.2f}, '
                           f'{math.degrees(best_pose[2]):.0f}°) '
                           f'{best_score}/{n_rays} '
                           f'(Δ {math.hypot(dxs, dys):.2f} m, '
                           f'{math.degrees(dyaw):+.0f}°)' + edge_msg)

    def start_continuous_localize(self):
        """Start a background thread that runs local_scan_match every ~1 s
        while SLAM is in localization mode. Keeps the bot's pose anchored
        to scan-vs-map matches continuously — without this, slam_toolbox's
        scan matcher drifts during motion and the live LiDAR points stop
        lining up with the saved map's walls."""
        if (self._continuous_localize_thread is not None
                and self._continuous_localize_thread.is_alive()):
            return
        self._continuous_localize_stop.clear()
        self._continuous_localize_thread = threading.Thread(
            target=self._continuous_localize_loop, daemon=True)
        self._continuous_localize_thread.start()
        _log('continuous_localize: started')

    def set_continuous_paused(self, paused):
        """Pause/resume the background continuous-localize loop (used by
        AutoNav while it drives — it owns localization for the run)."""
        self._continuous_paused = bool(paused)
        self._match_fail_streak = 0
        _log(f'continuous_localize: {"paused" if paused else "resumed"}')

    def stop_continuous_localize(self):
        self._continuous_localize_stop.set()
        t = self._continuous_localize_thread
        if t is not None:
            t.join(timeout=3.0)
        self._continuous_localize_thread = None
        _log('continuous_localize: stopped')

    def _continuous_localize_loop(self):
        INTERVAL_S = 1.0
        if self._continuous_localize_stop.wait(1.5):
            return
        last_predict_t = time.monotonic()
        while not self._continuous_localize_stop.is_set():
            if self.state != 'active' or self.mode != 'localization':
                if self._continuous_localize_stop.wait(INTERVAL_S):
                    return
                continue
            if self._continuous_paused:
                self._match_fail_streak = 0
                last_predict_t = time.monotonic()
                if self._continuous_localize_stop.wait(INTERVAL_S):
                    return
                continue
            # Spinning too fast to track: the 1 Hz ±25° search cannot follow
            # >~20°/s of rotation — matches during a fast arc EDGE-CLIP and
            # commit wrong yaws (and the failure streak then fires the ±180°
            # sweep mid-motion, which is how run 2's pose got corrupted).
            # Skip the cycle and let the IMU carry heading until it slows.
            omega = getattr(self, '_last_imu_omega_z', 0.0) or 0.0
            if abs(omega) > 0.35:
                self._match_fail_streak = 0
                last_predict_t = time.monotonic()
                if self._continuous_localize_stop.wait(0.3):
                    return
                continue
            if self._localize_in_progress.locked():
                if self._continuous_localize_stop.wait(INTERVAL_S):
                    return
                continue

            # Motion prediction: between scan-matches, integrate the
            # commanded forward velocity + IMU yaw rate to nudge self._pose
            # forward in the direction the bot is going. local_scan_match
            # then searches around this PREDICTED pose, not the stale
            # last-match pose. Without prediction, every fast-moving cycle
            # hit EDGE-CLIP because the bot moved further than the search
            # radius could find.
            now = time.monotonic()
            dt_pred = now - last_predict_t
            last_predict_t = now
            predicted = None
            try:
                predicted = self._predict_motion(dt_pred)
            except Exception as e:
                _log(f'predict_motion error: {e}')

            try:
                t0 = time.monotonic()
                # Pass prediction as the search prior. local_scan_match only
                # mutates self._pose on a successful match, so a failed
                # match no longer drifts the pose by the predicted amount.
                pose, msg = self.local_scan_match(prior_pose=predicted)
                dt_ms = (time.monotonic() - t0) * 1000.0
                if pose is not None and msg and 'no meaningful change' not in msg:
                    _log(f'continuous: {msg} ({dt_ms:.0f} ms)')

                # Escalating recovery. A streak of 'no good match' means the
                # bot is somewhere — or FACING somewhere — the ±25° local
                # search can't see. The classic case is a large yaw error
                # (e.g. the pose says 180° but the bot faces 0°): position
                # is roughly right, so first sweep ALL yaws at the current
                # spot (cheap), and only if that fails too fall back to a
                # full global_localize over the whole map.
                if pose is not None:
                    self._match_fail_streak = 0
                elif msg and 'no good match' in msg:
                    self._match_fail_streak += 1
                    if self._match_fail_streak == 4:
                        _log('continuous: 4 straight match failures — '
                             '±180° yaw sweep around current position')
                        rpose, rmsg = self.local_scan_match(
                            search_radius_m=0.30, search_yaw_deg=180.0,
                            step_m=0.10, step_deg=4.0,
                            min_hit_fraction=0.20)
                        _log(f'continuous: yaw sweep → {rmsg}')
                        if rpose is not None:
                            self._match_fail_streak = 0
                    elif self._match_fail_streak >= 7:
                        _log('continuous: still lost after yaw sweep — '
                             'running global_localize')
                        _gp, gmsg = self.global_localize()
                        _log(f'continuous: global_localize → {gmsg}')
                        self._match_fail_streak = 0
            except Exception as e:
                _log(f'continuous_localize error: {e}')
            if self._continuous_localize_stop.wait(INTERVAL_S):
                return

    def _predict_motion(self, dt):
        """Return a search-center pose for the next local_scan_match.

        YAW is predicted from the IMU gyro rate. x/y are predicted from the
        EKF's LiDAR-MEASURED forward velocity (lidar_odom). The original
        reason x/y prediction was removed — cmd_vx being an uncalibrated
        open-loop throttle guess — no longer applies: this velocity is
        measured from how far the walls actually moved between scans, and
        it reads ~0 when the bot isn't moving, so the icon-only-moves-on-
        evidence property is preserved. Without x/y prediction, a bot at
        0.5 m/s outruns the ±0.5 m search radius every cycle (the EDGE-CLIP
        logs) and the pose falls permanently behind.

        Frame note: self._pose yaw is matcher-frame (user yaw-offset baked
        in). The true map yaw is (theta - yaw_offset), and forward travel in
        the map runs LASER_NOSE_RAD further round because maps are built in
        the laser frame (92° mount, identity TF).

        The result is transient — never written back into self._pose."""
        if dt <= 0.0 or dt > 2.0:
            return None
        with self._lock:
            if self._pose is None:
                return None
            px, py, ptheta = self._pose
            offset = self._yaw_offset_rad
            vx = self._last_ekf_vx
            vx_ts = self._last_ekf_vx_ts
        # IMU yaw rate is accurate (gyro is reliable for short-term yaw).
        omega_z = getattr(self, '_last_imu_omega_z', 0.0) or 0.0
        new_theta = ptheta + omega_z * dt
        new_theta = (new_theta + math.pi) % (2 * math.pi) - math.pi
        if (time.monotonic() - vx_ts) > 0.5:
            vx = 0.0   # EKF stale/not running — don't extrapolate position
        heading = ptheta - offset + LASER_NOSE_RAD
        px += vx * dt * math.cos(heading)
        py += vx * dt * math.sin(heading)
        return (px, py, new_theta)

    def start(self, mode='mapping', map_file=''):
        mode = (mode or 'mapping').lower()
        if mode not in VALID_MODES:
            return False, f'invalid mode {mode!r}'
        map_file = (map_file or '').strip()
        if mode == 'localization':
            if not map_file:
                return False, 'localization requires a saved map'
            pg = os.path.join(MAP_DIR, map_file + '.posegraph')
            if not os.path.isfile(pg):
                return False, f'map not found: {pg}'
        with self._lock:
            if self.state not in ('idle', 'error'):
                return False, f'cannot start from {self.state}'
            self.state = 'starting'
            self.error = None
            self.mode = mode
            self.map_file = map_file
        threading.Thread(target=self._start_impl, daemon=True).start()
        return True, f'starting ({mode})'

    def _write_preview_sidecar(self, path):
        """Render the current /map to <path>.png + <path>.meta.json.

        Called after a successful save_map so the dashboard can show the
        saved map even when SLAM isn't running. The meta JSON also captures
        the bot's last-known pose so the preview can show "where you were
        when you saved" without re-running SLAM."""
        with self._lock:
            grid = self._map
            pose = self._pose
        if grid is None:
            return
        png, meta = self._render_grid_to_png(grid)
        if png is None or meta is None:
            return
        if pose is not None:
            meta['last_pose'] = {
                'x': pose[0], 'y': pose[1], 'theta': pose[2],
            }
        try:
            with open(path + '.png', 'wb') as f:
                f.write(png)
            import json
            with open(path + '.meta.json', 'w') as f:
                json.dump(meta, f)
        except OSError as e:
            _log(f'preview sidecar write failed: {e}')

    def set_lost_callback(self, fn):
        """Register a callback fired once when confidence drops to "lost".

        Used by the dashboard to soft-stop the driver if SLAM loses track
        while the bot is moving."""
        self._lost_callback = fn

    def jiggle_localize(self, duration_s=8.0, throttle=0.95):
        """Active-localization: rotate + nudge until scan-match locks.

        slam_toolbox in localization mode needs a few distinct scans before
        it can confidently match against the saved posegraph. Sitting still
        rarely works — every scan looks the same. A small motion sequence
        (left/right arcs + a forward nudge) gives the matcher enough
        parallax + rotational variation to find the correct location in
        the map.

        `throttle` defaults to 0.95 because this hardware has a stiff ESC
        deadband + sagging bypassed-battery voltage; anything below ~0.9
        merely warms the motor without breaking static friction. Lower it
        only if you know the bot can actually move at that command.

        While this runs, the "lost" callback is suspended: the very first
        successful match teleports the pose from the initial guess to the
        actual location, which looks like a huge anomalous jump to the
        confidence heuristic. We want that jump — it's the right answer.

        SAFETY: caller must ensure clear space ahead and to the sides. The
        routine does NOT do its own obstacle check. /stop and /drive (with
        a non-trivial command) both cancel the jiggle."""
        if self.state != 'active':
            return False, f'cannot jiggle: SLAM state={self.state}'
        if self._driver is None:
            return False, 'no driver'
        if self._jiggle_thread is not None and self._jiggle_thread.is_alive():
            return False, 'jiggle already running'
        duration_s = max(2.0, min(30.0, float(duration_s)))
        throttle = max(0.1, min(1.0, float(throttle)))
        self._jiggle_stop.clear()
        self._jiggle_status = 'running'
        self._jiggle_thread = threading.Thread(
            target=self._jiggle_run, args=(duration_s, throttle), daemon=True)
        self._jiggle_thread.start()
        return True, f'jiggling {duration_s:.0f}s @ throttle {throttle:.2f}'

    def stop_jiggle(self):
        """Cancel an in-progress jiggle. Safe to call when none is running."""
        if self._jiggle_thread is not None and self._jiggle_thread.is_alive():
            self._jiggle_stop.set()
            return True
        return False

    def is_jiggling(self):
        """True iff the jiggle thread is actively running."""
        return (self._jiggle_thread is not None
                and self._jiggle_thread.is_alive())

    def get_last_globalize_pose(self, max_age_s=30.0):
        """Return the most-recently-found global_localize pose if it's
        fresh, otherwise None. Used by AutoNav to skip a TF round-trip
        after publishing /initialpose."""
        with self._lock:
            pose = self._last_globalize_pose
            ts = self._last_globalize_ts
        if pose is None or (time.monotonic() - ts) > max_age_s:
            return None
        return pose

    def global_localize(self, sample_step_m=0.25, yaw_step_deg=5.0,
                        min_hit_fraction=0.18, refine=True):
        """Brute-force scan-vs-map search to find where the bot actually is.

        slam_toolbox's loop-closure only searches around its current pose
        estimate, so when that estimate is wrong (the default (0,0,0) when
        you're nowhere near where mapping started) it never finds the right
        match. This function does proper GLOBAL localization: for every
        free cell in the saved map and every rotation, project the current
        scan rays out from that pose and count how many fall on map walls.
        Highest hit count wins; publish to /initialpose so slam_toolbox
        adopts that as its pose estimate, then tracks from there.

        Cost on the Atom: ~150-300 free candidates × 45 yaws × 720 rays,
        fully numpy-vectorized at the inner level. About 1-3 seconds.

        Returns (pose, msg) on success, (None, error) otherwise."""
        try:
            import numpy as np
        except ImportError:
            return None, 'numpy unavailable'

        self._global_localize_status = 'running'
        with self._lock:
            grid = self._map
            scan = list(self._scan_points)

        if grid is None:
            self._global_localize_status = 'fail'
            return None, 'no saved map loaded yet (wait a moment after start)'
        if not scan or len(scan) < 30:
            self._global_localize_status = 'fail'
            return None, f'not enough scan points ({len(scan)})'

        w = grid.info.width
        h = grid.info.height
        res = grid.info.resolution
        ox = grid.info.origin.position.x
        oy = grid.info.origin.position.y

        # OccupancyGrid.data: -1 unknown, 0 free, 100 occupied. The matcher
        # only counts hits on *known* occupied cells; free / unknown / out-
        # of-bounds rays score zero.
        map_arr = np.asarray(grid.data, dtype=np.int8).reshape((h, w))
        occupied = (map_arr > 50)
        free = (map_arr == 0)

        # Scan rays in bot frame (x=forward, y=left). Drop rays that are
        # too close (probably the bot's own mount) or too far (range-noise).
        scan_xy = np.array([
            (d_mm * math.cos(math.radians(a_deg)) / 1000.0,
             d_mm * math.sin(math.radians(a_deg)) / 1000.0)
            for a_deg, d_mm in scan
            if 100.0 < d_mm < 5500.0
        ], dtype=np.float32)
        if len(scan_xy) < 30:
            self._global_localize_status = 'fail'
            return None, f'only {len(scan_xy)} usable rays — sensor blocked?'
        rx = scan_xy[:, 0]
        ry = scan_xy[:, 1]
        # Pre-rotate scan rays by -yaw_offset so the matcher operates in
        # the user-corrected frame. Match output IS physical yaw — no
        # post-match adjustment, no reversion possible.
        with self._lock:
            offset_rad = self._yaw_offset_rad
        if abs(offset_rad) > 1e-4:
            beta = -offset_rad
            cb = math.cos(beta); sb = math.sin(beta)
            new_rx = cb * rx - sb * ry
            new_ry = sb * rx + cb * ry
            rx, ry = new_rx, new_ry
        n_rays = len(rx)

        # Candidates: every free cell on a coarse-enough grid that the
        # search remains tractable. Skip walls/unknowns — the bot can't be
        # inside a wall.
        candidate_xys = []
        x_start = ox + res
        y_start = oy + res
        x_count = max(1, int((w * res - 2 * res) / sample_step_m))
        y_count = max(1, int((h * res - 2 * res) / sample_step_m))
        for i in range(x_count):
            cx = x_start + i * sample_step_m
            mx = int((cx - ox) / res)
            if not (0 <= mx < w):
                continue
            for j in range(y_count):
                cy = y_start + j * sample_step_m
                my = int((cy - oy) / res)
                if not (0 <= my < h):
                    continue
                if free[my, mx]:
                    candidate_xys.append((cx, cy))
        if not candidate_xys:
            self._global_localize_status = 'fail'
            return None, 'no free cells in saved map'

        candidate_xys = np.array(candidate_xys, dtype=np.float32)
        cx_arr = candidate_xys[:, 0]   # (C,)
        cy_arr = candidate_xys[:, 1]
        yaws = np.radians(
            np.arange(-180.0, 180.0, yaw_step_deg, dtype=np.float32))

        _log(f'global_localize: {len(cx_arr)} positions × {len(yaws)} yaws '
             f'× {n_rays} rays')
        t0 = time.monotonic()

        best_score = -1
        best_pose = None
        # Second-best tracker for ambiguity detection — if top-1 is barely
        # better than top-2, the match is ambiguous and we should fail.
        runner_up_score = -1
        x_hi = ox + w * res
        y_hi = oy + h * res
        # Intermediate-free check fractions: each scan ray's path from bot
        # to its measured endpoint must pass through FREE map cells.
        # Without this, the matcher accepts a pose where the ray endpoint
        # happens to land on a wall but the ray would have been physically
        # blocked by another wall en route — exactly the failure mode where
        # the bot is in a small room but global_localize places it in a
        # corridor (corridor walls catch some ray endpoints by coincidence).
        free_or_unknown = ~occupied  # cells that don't block rays
        check_fractions = (0.35, 0.65, 0.90)
        for yaw in yaws:
            cos_y = float(np.cos(yaw))
            sin_y = float(np.sin(yaw))
            # World-frame ray deltas (the ray endpoints relative to bot).
            dx = cos_y * rx - sin_y * ry        # (R,)
            dy = sin_y * rx + cos_y * ry
            # Endpoint world coords (C, R) and occupancy check
            wx = cx_arr[:, None] + dx[None, :]
            wy = cy_arr[:, None] + dy[None, :]
            in_bounds = ((wx >= ox) & (wx < x_hi) &
                         (wy >= oy) & (wy < y_hi))
            mx = np.clip(((wx - ox) / res).astype(np.int32), 0, w - 1)
            my = np.clip(((wy - oy) / res).astype(np.int32), 0, h - 1)
            endpoint_occ = occupied[my, mx] & in_bounds

            # Intermediate-free: at each sampled fraction along the ray,
            # the cell must NOT be a wall. (Unknown cells are tolerated —
            # the map may have gaps, but a known wall en route is fatal.)
            inter_clear = np.ones_like(endpoint_occ)
            for f in check_fractions:
                fwx = cx_arr[:, None] + f * dx[None, :]
                fwy = cy_arr[:, None] + f * dy[None, :]
                fin = ((fwx >= ox) & (fwx < x_hi) &
                       (fwy >= oy) & (fwy < y_hi))
                fmx = np.clip(((fwx - ox) / res).astype(np.int32),
                              0, w - 1)
                fmy = np.clip(((fwy - oy) / res).astype(np.int32),
                              0, h - 1)
                inter_clear &= (free_or_unknown[fmy, fmx] & fin)

            good_ray = endpoint_occ & inter_clear     # (C, R)
            scores = good_ray.sum(axis=1)             # (C,)
            c_best = int(scores.argmax())
            top_score = int(scores[c_best])
            if top_score > best_score:
                runner_up_score = best_score  # demote prior best
                best_score = top_score
                best_pose = (float(cx_arr[c_best]),
                             float(cy_arr[c_best]),
                             float(yaw))
            elif top_score > runner_up_score:
                runner_up_score = top_score

        dt = time.monotonic() - t0
        if best_pose is None:
            self._global_localize_status = 'fail'
            return None, 'no match found'
        _log(f'global_localize: coarse best ({best_pose[0]:.2f}, '
             f'{best_pose[1]:.2f}, {math.degrees(best_pose[2]):.0f}°) '
             f'score {best_score}/{n_rays} in {dt:.2f}s')

        # Refinement pass: at 1° step around the top yaw and 10 cm step
        # around the top position. This catches near-symmetric cases
        # where the coarse 5° search picked an OK-but-not-best yaw —
        # rejecting wrong-by-90° kinds of matches.
        if refine and len(yaws) > 1:
            t1 = time.monotonic()
            cx0, cy0, yaw0 = best_pose
            fine_xs = np.arange(cx0 - 0.30, cx0 + 0.31, 0.05, dtype=np.float32)
            fine_ys = np.arange(cy0 - 0.30, cy0 + 0.31, 0.05, dtype=np.float32)
            fine_yaws = np.radians(np.arange(
                math.degrees(yaw0) - 6.0,
                math.degrees(yaw0) + 6.001,
                1.0, dtype=np.float32))
            for cx in fine_xs:
                mx0 = int((cx - ox) / res)
                if not (0 <= mx0 < w):
                    continue
                for cy in fine_ys:
                    my0 = int((cy - oy) / res)
                    if not (0 <= my0 < h):
                        continue
                    if not free[my0, mx0]:
                        continue
                    for yaw in fine_yaws:
                        cos_y = float(np.cos(yaw))
                        sin_y = float(np.sin(yaw))
                        dx_r = cos_y * rx - sin_y * ry
                        dy_r = sin_y * rx + cos_y * ry
                        wxr = cx + dx_r
                        wyr = cy + dy_r
                        inb = ((wxr >= ox) & (wxr < x_hi) &
                               (wyr >= oy) & (wyr < y_hi))
                        mxr = np.clip(((wxr - ox) / res).astype(np.int32),
                                      0, w - 1)
                        myr = np.clip(((wyr - oy) / res).astype(np.int32),
                                      0, h - 1)
                        end_occ = occupied[myr, mxr] & inb
                        # Same intermediate-free check as the coarse pass.
                        inter = np.ones_like(end_occ)
                        for f in check_fractions:
                            fwx = cx + f * dx_r
                            fwy = cy + f * dy_r
                            fin = ((fwx >= ox) & (fwx < x_hi) &
                                   (fwy >= oy) & (fwy < y_hi))
                            fmx = np.clip(((fwx - ox) / res).astype(np.int32),
                                          0, w - 1)
                            fmy = np.clip(((fwy - oy) / res).astype(np.int32),
                                          0, h - 1)
                            inter &= (free_or_unknown[fmy, fmx] & fin)
                        hits = (end_occ & inter).sum()
                        if hits > best_score:
                            best_score = int(hits)
                            best_pose = (float(cx), float(cy), float(yaw))
            _log(f'global_localize: refined to ({best_pose[0]:.2f}, '
                 f'{best_pose[1]:.2f}, {math.degrees(best_pose[2]):.0f}°) '
                 f'score {best_score}/{n_rays} '
                 f'in extra {time.monotonic() - t1:.2f}s')

        hit_frac = best_score / n_rays
        if hit_frac < min_hit_fraction:
            self._global_localize_status = 'fail'
            return None, (f'best match too weak: {best_score}/{n_rays} = '
                          f'{hit_frac*100:.0f}% (need >={min_hit_fraction*100:.0f}%) '
                          'saved map may be incomplete')

        # Ambiguity check: if the runner-up coarse score is within 15% of
        # the best, the match is ambiguous — better to fail and re-run
        # (after a jiggle) than commit to a possibly-wrong yaw.
        # Margin bumped from 15% to 25% to compensate for the lower
        # min_hit_fraction (raytrace scoring scores correct matches in the
        # 20-30% range, so we lean on the ambiguity check more heavily).
        if runner_up_score > 0:
            margin = (best_score - runner_up_score) / best_score
            _log(f'global_localize: best {best_score}, runner-up {runner_up_score}, '
                 f'margin {margin*100:.0f}%')
            if margin < 0.25:
                self._global_localize_status = 'fail'
                return None, (f'ambiguous match: best {best_score} vs runner-up '
                              f'{runner_up_score} (margin {margin*100:.0f}%) — '
                              'multiple poses fit similarly. Move the bot or '
                              'jiggle for more disambiguating scans.')

        # NOTE (removed June 2026): a "first-localize auto-yaw-offset" hack
        # used to live here — it overrode the matcher's yaw with the IMU's
        # startup yaw (always ~0°) and persisted the difference as a yaw
        # offset. Its premise was wrong (the bot rarely starts facing the
        # map's zero direction), so every new session baked the bot's
        # arbitrary starting orientation into the offset and COMPOUNDED it
        # across sessions (logged history swung +1° → -174° → +15°). That
        # rotated the displayed pose off the walls and biased every AutoNav
        # heading decision by the offset (positions are true-map-frame,
        # yaws were offset-frame). The matcher's yaw IS the truth — publish
        # it. yaw_offset remains a purely MANUAL slider correction.
        adjusted_pose = best_pose

        ok = self._publish_initial_pose(*adjusted_pose)
        if not ok:
            self._global_localize_status = 'fail'
            return None, 'publish to /initialpose failed'
        self._global_localize_status = 'success'
        # Cache the just-localized (offset-corrected) pose so callers can
        # use it without waiting for slam_toolbox's TF tree to absorb
        # /initialpose. Also update self._pose AND pin it — without the
        # pin the next TF lookup would revert this correction.
        with self._lock:
            self._last_globalize_pose = adjusted_pose
            self._last_globalize_ts = time.monotonic()
            self._last_globalize_hit_frac = hit_frac
            self._pose = adjusted_pose
            self._pose_ts = time.monotonic()
            self._pose_pin_until_ts = time.monotonic() + self.POSE_PIN_S
        return adjusted_pose, (f'localized at ({adjusted_pose[0]:.2f}, '
                               f'{adjusted_pose[1]:.2f}, '
                               f'{math.degrees(adjusted_pose[2]):.0f}°), '
                               f'hits {best_score}/{n_rays} ({hit_frac*100:.0f}%)')

    def _publish_initial_pose(self, x, y, yaw):
        """Publish PoseWithCovarianceStamped on /initialpose so slam_toolbox
        adopts the given pose. Safe to call from any thread — rclpy
        publishers are thread-safe."""
        if self._initialpose_pub is None:
            _log('_publish_initial_pose: publisher not ready')
            return False
        try:
            from geometry_msgs.msg import PoseWithCovarianceStamped
            # `yaw` is this module's LASER-frame yaw; /initialpose carries a
            # BASE pose. With the truthful -92° base->laser TF, slam_toolbox
            # derives sensor yaw = base yaw - 92°, so publish base = laser + 92°
            # and it lands back on exactly the laser yaw the matcher found.
            base_yaw = yaw + LASER_NOSE_RAD
            msg = PoseWithCovarianceStamped()
            msg.header.frame_id = 'map'
            msg.pose.pose.position.x = float(x)
            msg.pose.pose.position.y = float(y)
            msg.pose.pose.position.z = 0.0
            msg.pose.pose.orientation.z = float(math.sin(base_yaw / 2.0))
            msg.pose.pose.orientation.w = float(math.cos(base_yaw / 2.0))
            # 6×6 covariance diagonal: trust our pose moderately. AMCL
            # default is [0.25, 0.25, 0.06853] for (x, y, yaw).
            cov = [0.0] * 36
            cov[0] = 0.10    # x variance (m²)
            cov[7] = 0.10    # y variance
            cov[35] = 0.05   # yaw variance (rad²)
            msg.pose.covariance = cov
            self._initialpose_pub.publish(msg)
            _log(f'/initialpose published: ({x:.2f}, {y:.2f}, '
                 f'{math.degrees(yaw):.0f}°)')
            return True
        except Exception as e:
            _log(f'_publish_initial_pose failed: {e}')
            return False

    def _jiggle_run(self, duration_s, throttle):
        # Throttle defaults to 0.95 — anything less doesn't actually move the
        # bot on this hardware (ESC deadband + bypassed-battery torque sag).
        # At 0.95 the bot moves fast, so segments are short to keep total
        # displacement bounded. Pauses between segments let momentum bleed
        # before the next direction. All-forward motion — no reverse, which
        # would trigger brake-priming and add complexity. (angle, dur_s)
        steer_segments = [
            (-1.0, 0.30),   # arc right
            ( 0.0, 0.20),   # settle
            ( 1.0, 0.40),   # arc left (longer to swing past)
            ( 0.0, 0.20),
            (-1.0, 0.40),   # arc right (longer to swing back)
            ( 0.0, 0.20),
            ( 1.0, 0.30),   # arc left to roughly center
            ( 0.0, 0.25),
            ( 0.0, 0.30),   # straight forward nudge for parallax
            ( 0.0, 0.30),   # final settle
        ]
        # Build the active-throttle list: even-indexed segments drive at
        # `throttle`, odd-indexed (pause) at 0. The straight-forward segment
        # at index 8 also drives.
        segments = [
            (a, throttle if i % 2 == 0 else 0.0, dur)
            for i, (a, dur) in enumerate(steer_segments)
        ]

        # Suspend the lost callback so the initial localization snap (which
        # looks like a huge unexplained pose jump) doesn't trip the auto-stop.
        with self._lock:
            saved_cb = self._lost_callback
            self._lost_callback = None

        start = time.monotonic()
        seg_idx = 0
        seg_start = start
        loops = 0
        try:
            while not self._jiggle_stop.is_set():
                now = time.monotonic()
                if now - start >= duration_s:
                    self._jiggle_status = 'done'
                    break
                angle, throttle, dur = segments[seg_idx]
                if now - seg_start >= dur:
                    seg_idx = (seg_idx + 1) % len(segments)
                    if seg_idx == 0:
                        loops += 1
                    seg_start = now
                    angle, throttle, dur = segments[seg_idx]
                try:
                    self._driver.set_cmd(angle, throttle)
                except Exception as e:
                    _log(f'jiggle drive failed: {e}')
                    self._jiggle_status = 'error'
                    break
                time.sleep(0.05)
            if self._jiggle_stop.is_set() and self._jiggle_status == 'running':
                self._jiggle_status = 'cancelled'
        finally:
            # Always neutralize the driver and reset confidence state — the
            # large pose jump from localization will otherwise leave stale
            # entries in _pose_history that immediately re-trip lost.
            try:
                self._driver.set_cmd(0.0, 0.0)
            except Exception:
                pass
            with self._lock:
                self._lost_callback = saved_cb
                self._pose_history = []
                self._was_lost = False
                self._confidence = 1.0
            _log(f'jiggle {self._jiggle_status} (loops={loops})')

    def _auto_localize_after_start(self):
        """Localization startup: global_localize first, then a short jiggle.

        Sequence:
          1. Brief grace so slam_toolbox publishes /map and we receive scans.
          2. global_localize: brute-force scan-vs-map search to FIND where
             the bot actually is, then publish to /initialpose so
             slam_toolbox adopts that pose. This solves the case slam_toolbox
             alone can't: the bot isn't near (0,0,0) so loop-closure has
             nothing to find within its 5 m search radius.
          3. Short jiggle for verification — varied scans let slam_toolbox's
             continuous tracking lock onto the seeded pose.

        Lost-track callback was suspended at state='active' by _start_impl;
        restored here on exit."""
        GRACE_S = 3.0
        # Shorter jiggle now that we seed the pose with global_localize —
        # we don't need a long motion to "find" anything, just to verify the
        # seeded pose tracks.
        JIGGLE_S = 4.0
        try:
            # 1) Grace — wait for /map and a usable scan
            deadline = time.monotonic() + GRACE_S
            while time.monotonic() < deadline:
                if self.state != 'active':
                    return
                with self._lock:
                    have_map = self._map is not None
                    have_scan = len(self._scan_points) > 30
                if have_map and have_scan:
                    break
                time.sleep(0.25)

            if self.state != 'active':
                return

            # 2) Global localization — find the bot in the saved map.
            # If the first attempt has only modest confidence (the case
            # where the corridor's rotational symmetry confuses the
            # matcher), jiggle first to get varied scans and try again.
            # With raytrace-consistent scoring, a "good" match typically
            # scores 35-50% of rays — anything below 40 % gets retried.
            # Raytrace-consistent scoring tops out ~25-35% for correct
            # matches (every ray must pass endpoint + 3 intermediate checks).
            # 0.25 means "if we cleared the min_hit threshold (0.18) but
            # not by much, jiggle and retry for varied scans".
            CONFIDENT_HIT_FRAC = 0.25
            _log('auto-localize: running global_localize')
            pose, msg = self.global_localize()
            _log(f'auto-localize: global_localize → {msg}')
            initial_hit_frac = self._last_globalize_hit_frac
            initial_pose = pose

            # 3) Jiggle (always — gives varied scans for both verifying
            # and potentially re-localizing if the first match was iffy).
            if pose is not None:
                time.sleep(1.5)
            if self.state != 'active':
                return
            if self._jiggle_thread is not None and self._jiggle_thread.is_alive():
                _log('auto-localize: another routine moving, joining it')
                self._jiggle_thread.join()
                return

            _log('auto-localize: starting verification jiggle')
            ok, msg = self.jiggle_localize(duration_s=JIGGLE_S)
            if not ok:
                _log(f'auto-localize: jiggle could not start ({msg})')
                return
            t = self._jiggle_thread
            if t is not None:
                t.join()

            # 4) If the initial localization was iffy, run global_localize
            # again now that the bot has moved a bit and scans are varied.
            if self.state != 'active':
                return
            if initial_pose is None or initial_hit_frac < CONFIDENT_HIT_FRAC:
                _log(f'auto-localize: initial hit_frac '
                     f'{initial_hit_frac*100:.0f}% < {CONFIDENT_HIT_FRAC*100:.0f}%'
                     ' — re-globalizing after jiggle for disambiguation')
                time.sleep(0.5)  # let scan settle after jiggle
                pose2, msg2 = self.global_localize()
                _log(f'auto-localize: re-globalize → {msg2}')
        finally:
            with self._lock:
                saved = getattr(self, '_localize_startup_saved_cb', None)
                self._lost_callback = saved
                self._pose_history = []
                self._confidence = 1.0
                self._was_lost = False
                if hasattr(self, '_localize_startup_saved_cb'):
                    delattr(self, '_localize_startup_saved_cb')
            _log('auto-localize: phase complete, lost-track callback restored')
            # Kick off continuous local_scan_match now that initial
            # localization is settled. Keeps pose anchored during driving.
            self.start_continuous_localize()

    def _update_confidence(self, new_pose, now):
        """Recompute pose confidence from UNEXPLAINED motion magnitudes.

        Earlier version simply summed inter-sample distances, which treated
        legitimate forward driving as "jumpy" — every scan-match correction
        during normal motion contributed to the jump count, and the bot
        got flagged "lost" whenever it actually moved. New version compares
        each pose-step displacement against what the driver commanded over
        that step (cmd_vx integrated over dt). Only the residual — the
        scan-match snap on top of commanded motion — counts toward "lost".

        Tolerances:
          * 2× cmd_vx headroom — vel_factor calibration is never perfect.
          * "Big anomaly" = >0.3 m of unexplained displacement in <0.4 s.
          * Stationary check is still strict: if cmd_vx ≈ 0 and the pose
            jumps, that IS suspicious and we count it."""
        fire_callback = False
        with self._lock:
            # Expected-motion allowance: whichever is larger of the open-loop
            # throttle estimate and the EKF's LiDAR-measured speed. The
            # measured value covers the cases throttle gets wrong (coasting,
            # being pushed, sag) without loosening the stationary check.
            cmd_vx = max(abs(self._last_cmd_vx), abs(self._last_ekf_vx))
            self._pose_history.append(
                (new_pose[0], new_pose[1], new_pose[2], now, cmd_vx))
            cutoff = now - 2.0
            while self._pose_history and self._pose_history[0][3] < cutoff:
                self._pose_history.pop(0)
            if len(self._pose_history) < 3:
                return
            anom_total = 0.0
            big_anom = 0
            for i in range(1, len(self._pose_history)):
                # Tolerate older entries that lack cmd_vx (right after the
                # patch loads — list mixes 4-tuples and 5-tuples briefly).
                p1 = self._pose_history[i - 1]
                p2 = self._pose_history[i]
                x1, y1, _, t1 = p1[0], p1[1], p1[2], p1[3]
                x2, y2, _, t2 = p2[0], p2[1], p2[2], p2[3]
                cv1 = p1[4] if len(p1) > 4 else 0.0
                d = math.hypot(x2 - x1, y2 - y1)
                dt = t2 - t1
                if dt <= 0:
                    continue
                # 2× cmd_vx allowance for vel_factor miscal + brief
                # acceleration. e.g. cmd_vx=0.5 over 100 ms → expect ≤10 cm.
                expected = abs(cv1) * dt * 2.0
                anom = max(0.0, d - expected)
                anom_total += anom
                if dt < 0.4 and anom > 0.3:
                    big_anom += 1
            c = 1.0
            # Accumulated unexplained motion: >30 cm over 2 s starts to bite.
            if anom_total > 0.3:
                c -= min(0.5, (anom_total - 0.3) * 1.0)
            if big_anom > 0:
                c -= 0.4 * big_anom
            c = max(0.0, c)
            self._confidence = c
            is_lost = c < 0.25
            if is_lost and not self._was_lost:
                fire_callback = self._lost_callback is not None
            self._was_lost = is_lost
        if fire_callback:
            try:
                self._lost_callback()
            except Exception as e:
                _log(f'lost_callback raised: {e}')

    def list_maps(self):
        """Return saved map basenames (each has a matching .posegraph + .data)."""
        try:
            names = []
            for p in sorted(glob.glob(os.path.join(MAP_DIR, '*.posegraph'))):
                base = os.path.basename(p)[:-len('.posegraph')]
                names.append({
                    'name': base,
                    'size_bytes': os.path.getsize(p),
                    'mtime': os.path.getmtime(p),
                })
            return names
        except OSError:
            return []

    def stop(self):
        with self._lock:
            if self.state not in ('active', 'starting', 'error'):
                return False, f'cannot stop from {self.state}'
            self.state = 'stopping'
        threading.Thread(target=self._stop_impl, daemon=True).start()
        return True, 'stopping'

    def status(self):
        now = time.monotonic()
        with self._lock:
            out = {
                'state': self.state,
                'error': self.error,
                'mode': self.mode,
                'map_file': self.map_file,
                'scan': {
                    'available': self._scan_ts > 0,
                    'rate_hz': round(self._scan_rate_hz, 1),
                    'age_ms': int((now - self._scan_ts) * 1000) if self._scan_ts else None,
                    'count': len(self._scan_points),
                },
                'map': None,
                'pose': None,
                'jiggle': self._jiggle_status,
                'global_localize': self._global_localize_status,
            }
            if self._map is not None:
                info = self._map.info
                total = info.width * info.height
                known = self._map_known_cells
                cell_m2 = info.resolution * info.resolution
                out['map'] = {
                    'w': info.width, 'h': info.height,
                    'res': round(info.resolution, 4),
                    'ox': round(info.origin.position.x, 3),
                    'oy': round(info.origin.position.y, 3),
                    'age_ms': int((now - self._map_ts) * 1000) if self._map_ts else None,
                    'known_cells': known,
                    'occupied_cells': self._map_occupied_cells,
                    'total_cells': total,
                    'coverage_pct': round(known / total * 100, 1) if total else 0.0,
                    'explored_m2': round(known * cell_m2, 2),
                }
            out['confidence'] = round(self._confidence, 3)
            out['lost'] = self._was_lost
            # Odometry diagnostics: what we're commanding vs what imu_node has integrated.
            out['odom'] = {
                'cmd_vx': round(self._last_cmd_vx, 4),  # last linear.x we published
                'x': round(self._odom_xy[0], 4) if self._odom_xy else None,
                'y': round(self._odom_xy[1], 4) if self._odom_xy else None,
                'yaw_deg': round(math.degrees(self._odom_yaw), 1),
                'age_ms': int((now - self._odom_ts) * 1000) if self._odom_ts else None,
            }
            out['imu_diag'] = {
                'count': getattr(self, '_imu_count', 0),
                'omega_z': round(getattr(self, '_last_imu_omega_z', 0.0), 4),
                'age_ms': int((now - getattr(self, '_last_imu_ts', 0.0)) * 1000)
                           if getattr(self, '_last_imu_ts', 0.0) else None,
            }
            if self._pose is not None:
                x, y, th = self._pose
                out['pose'] = {
                    'x': round(x, 3), 'y': round(y, 3),
                    'theta': round(th, 4),
                    'age_ms': int((now - self._pose_ts) * 1000) if self._pose_ts else None,
                }
        return out

    def scan_snapshot(self):
        now = time.monotonic()
        with self._lock:
            pts = self._scan_points
            ts = self._scan_ts
            rate = self._scan_rate_hz
        age = (now - ts) * 1000 if ts else None
        return pts, age, rate

    @staticmethod
    def _render_grid_to_png(grid):
        """Pure function: OccupancyGrid -> (png_bytes, meta_dict).

        Used both for the live `/slam/map.png` endpoint and to persist a
        viewable snapshot when the user saves a map."""
        if grid is None:
            return None, None
        try:
            import numpy as np
            import cv2
        except ImportError as e:
            _log(f'map render: numpy/cv2 missing: {e}')
            return None, None
        w, h = grid.info.width, grid.info.height
        if w == 0 or h == 0:
            return None, None
        arr_raw = bytes(bytearray((b & 0xFF) for b in grid.data))
        arr = np.frombuffer(arr_raw, dtype=np.int8).reshape((h, w))
        img = np.full((h, w), 127, dtype=np.uint8)  # unknown -> mid gray
        img[arr == 0] = 255   # free -> white
        img[arr > 50] = 0     # occupied -> black
        img = np.flipud(img)
        ok, buf = cv2.imencode('.png', img)
        if not ok:
            return None, None
        meta = {
            'w': w, 'h': h,
            'res': grid.info.resolution,
            'ox': grid.info.origin.position.x,
            'oy': grid.info.origin.position.y,
        }
        return buf.tobytes(), meta

    def map_png_bytes(self):
        """Render the latest occupancy grid as PNG bytes + metadata dict."""
        with self._lock:
            m = self._map
        return self._render_grid_to_png(m)

    def save_map(self, name):
        """Save the current map via slam_toolbox's serialize_pose_graph service.

        Success is detected by checking whether the .posegraph file was actually
        written within the last few seconds — way more reliable than parsing the
        ros2 service call's stdout (which has changed format between versions
        and gave us a "save failed" alert even when the file was successfully
        written)."""
        with self._lock:
            if self.state != 'active':
                return False, f'SLAM not active (state={self.state})'
            if self.mode == 'localization':
                return False, 'save_map disabled in localization mode'
        safe = ''.join(c for c in name if c.isalnum() or c in '-_')
        if not safe:
            return False, 'invalid name'
        os.makedirs(MAP_DIR, exist_ok=True)
        path = os.path.join(MAP_DIR, safe)
        pg_path = path + '.posegraph'
        # Note the file's pre-call state so we can detect "fresh write".
        pre_mtime = os.path.getmtime(pg_path) if os.path.isfile(pg_path) else 0.0
        cmd = ('source ' + ROS_SETUP + ' && '
               'ros2 service call /slam_toolbox/serialize_map '
               'slam_toolbox/srv/SerializePoseGraph '
               '"{filename: \'' + path + '\'}"')
        try:
            r = subprocess.run(['bash', '-c', cmd], capture_output=True,
                               text=True, timeout=20)
            out = (r.stdout + r.stderr).strip()
            # Truth-source: is the file on disk now, and was it freshly written?
            if os.path.isfile(pg_path):
                new_mtime = os.path.getmtime(pg_path)
                if new_mtime > pre_mtime:
                    size_mb = os.path.getsize(pg_path) / 1e6
                    # Also persist a viewable PNG snapshot + metadata so the
                    # saved map can be previewed in the dashboard without ever
                    # re-launching slam_toolbox.
                    self._write_preview_sidecar(path)
                    _log(f'saved {pg_path} ({size_mb:.1f} MB)')
                    return True, f'saved {safe}.posegraph ({size_mb:.1f} MB)'
            # File not written — return the service output so the user can see why
            _log(f'save_map failed: rc={r.returncode} out={out[:200]}')
            return False, f'rc={r.returncode}: {out[:300]}'
        except subprocess.TimeoutExpired:
            return False, 'service call timed out (slam_toolbox unresponsive)'
        except Exception as e:
            return False, str(e)

    def _start_impl(self):
        try:
            _log('starting: pausing serial lidar')
            if self._serial_lidar is not None:
                try:
                    self._serial_lidar.pause()
                except Exception as e:
                    _log(f'serial lidar pause failed: {e}')
            time.sleep(0.6)

            _log('clearing /dev/shm/fastrtps_*')
            self._clear_shm()

            _log('resetting RPLidar')
            self._reset_rplidar()

            with self._lock:
                mode = self.mode
                map_file = self.map_file
            map_path = os.path.join(MAP_DIR, map_file) if map_file else ''
            _log(f'spawning ros2 launch mode={mode} map={map_path!r} '
                 '(log: /tmp/slam_launch.log)')
            env = os.environ.copy()
            env['SLAM_MODE'] = mode
            env['SLAM_MAP_FILE'] = map_path
            env['RMW_IMPLEMENTATION'] = 'rmw_cyclonedds_cpp'
            env['CYCLONEDDS_URI'] = 'file:///home/deepracer/cyclonedds.xml'
            launch_log = open('/tmp/slam_launch.log', 'wb')
            self._launch_proc = subprocess.Popen(
                ['bash', '-c',
                 'set +u; source ' + ROS_SETUP +
                 ' && source ' + RF2O_WS_SETUP +
                 ' && exec ros2 launch ' + LAUNCH_FILE],
                stdout=launch_log, stderr=subprocess.STDOUT,
                preexec_fn=os.setpgrp,
                env=env,
            )

            self._ros_stop.clear()
            self._scan_window_start = time.monotonic()
            self._ros_thread = threading.Thread(target=self._ros_spin, daemon=True)
            self._ros_thread.start()

            # Success criterion differs by mode:
            #   localization → /map (deserialized from the saved graph) is
            #                  enough; you can start without the LiDAR and
            #                  see where the bot thinks it is even before
            #                  scans start flowing. Bigger saved maps take
            #                  longer to build the initial occupancy grid.
            #   mapping / lifelong → /scan is required (no scans = no map
            #                  ever gets built).
            #
            # We poll instead of sleeping a fixed interval — that way fast
            # starts go active in ~5s and slow ones get up to MAX_WAIT.
            MAX_WAIT = 30.0
            POLL = 0.5
            with self._lock:
                mode = self.mode
            elapsed = 0.0
            started = False
            while elapsed < MAX_WAIT and not self._ros_stop.is_set():
                time.sleep(POLL)
                elapsed += POLL
                with self._lock:
                    got_scan = self._scan_ts > 0
                    got_map = self._map is not None
                if mode == 'localization':
                    if got_map:
                        started = True
                        break
                else:
                    if got_scan:
                        started = True
                        break

            with self._lock:
                got_scan = self._scan_ts > 0
                got_map = self._map is not None
            if started:
                # Race-fix: suspend the lost-track callback BEFORE flipping
                # state to active. slam_toolbox's first scan-match snap in
                # localization mode looks like a huge anomalous jump to the
                # confidence heuristic and would otherwise fire driver.stop()
                # before the auto-localize thread can even start. Stored on
                # self and restored at the end of _auto_localize_after_start.
                if mode == 'localization':
                    with self._lock:
                        self._localize_startup_saved_cb = self._lost_callback
                        self._lost_callback = None
                        self._pose_history = []
                        self._confidence = 1.0
                        self._was_lost = False
                with self._lock:
                    self.state = 'active'
                _log(f'active ({mode}, after {elapsed:.1f}s, scan={got_scan} map={got_map})')
                if mode == 'localization':
                    threading.Thread(
                        target=self._auto_localize_after_start,
                        daemon=True).start()
            else:
                if mode == 'localization':
                    err_msg = (f'localization: /map not published within {MAX_WAIT:.0f}s — '
                               'is the saved map present + valid? see /tmp/slam_launch.log')
                else:
                    err_msg = (f'no scan received within {MAX_WAIT:.0f}s of launch — is '
                               'the LiDAR plugged in? see /tmp/slam_launch.log')
                _log(f'{mode} start failed: {err_msg}')
                self._teardown_after_failure()
                with self._lock:
                    self.state = 'error'
                    self.error = err_msg
        except Exception as e:
            _log(f'start failed: {e}')
            try: self._teardown_after_failure()
            except Exception: pass
            with self._lock:
                self.state = 'error'
                self.error = str(e)

    def _teardown_after_failure(self):
        """Best-effort cleanup so the next /slam/start can succeed."""
        self.stop_continuous_localize()
        self._ros_stop.set()
        if self._ros_thread is not None:
            self._ros_thread.join(timeout=3)
            self._ros_thread = None
        if self._launch_proc is not None and self._launch_proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._launch_proc.pid), signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                self._launch_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self._launch_proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self._launch_proc = None
        for name in ('async_slam_toolbox_node', 'lifelong_slam_toolbox_node',
                     'localization_slam_toolbox_node', 'rplidar_composition',
                     'static_transform_publisher', 'scan_filter_node.py',
                     'imu_node.py', 'rf2o_laser_odometry_node', 'ekf_node',
                     'cmd_vel_bridge.py', 'lidar_odom.py'):
            subprocess.run(['pkill', '-9', '-f', name],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.5)
        self._clear_shm()
        self._reset_rplidar()
        if self._serial_lidar is not None:
            try:
                self._serial_lidar.resume()
            except Exception:
                pass
        # Restore lost-track callback in case failure happened mid-startup.
        with self._lock:
            if hasattr(self, '_localize_startup_saved_cb'):
                self._lost_callback = self._localize_startup_saved_cb
                delattr(self, '_localize_startup_saved_cb')

    def _stop_impl(self):
        try:
            # Each shutdown step wrapped so one failure (e.g. an attribute
            # missing on a half-initialised SlamManager) doesn't strand the
            # rest — that's the bug that caused "couldn't stop SLAM".
            try:
                self.stop_continuous_localize()
            except Exception as e:
                _log(f'stop_continuous_localize failed: {e}')
            _log('stopping ROS spinner')
            try:
                self._ros_stop.set()
                if self._ros_thread is not None:
                    self._ros_thread.join(timeout=3)
                    self._ros_thread = None
            except Exception as e:
                _log(f'ros spin stop failed: {e}')

            if self._launch_proc is not None and self._launch_proc.poll() is None:
                _log('SIGINT launch group')
                try:
                    os.killpg(os.getpgid(self._launch_proc.pid), signal.SIGINT)
                except ProcessLookupError:
                    pass
                try:
                    self._launch_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    _log('SIGKILL launch group')
                    try:
                        os.killpg(os.getpgid(self._launch_proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        self._launch_proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
            self._launch_proc = None

            for name in ('async_slam_toolbox_node', 'lifelong_slam_toolbox_node',
                         'localization_slam_toolbox_node', 'rplidar_composition',
                         'static_transform_publisher', 'scan_filter_node.py',
                         'imu_node.py', 'rf2o_laser_odometry_node', 'ekf_node',
                         'cmd_vel_bridge.py', 'lidar_odom.py'):
                subprocess.run(['pkill', '-9', '-f', name],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1)

            _log('clearing SHM + resetting lidar')
            self._clear_shm()
            self._reset_rplidar()

            with self._lock:
                self._scan_points = []
                self._scan_ts = 0
                self._scan_rate_hz = 0
                self._map = None
                self._map_ts = 0
                self._map_known_cells = 0
                self._map_occupied_cells = 0
                self._pose = None
                self._pose_ts = 0
                self._pose_history = []
                self._confidence = 1.0
                self._was_lost = False
                # If stop interrupted the localization startup phase, the
                # lost-track callback is still suspended. Restore it so the
                # safety works on the next start, regardless of whether
                # auto-localize ran to completion.
                if hasattr(self, '_localize_startup_saved_cb'):
                    self._lost_callback = self._localize_startup_saved_cb
                    delattr(self, '_localize_startup_saved_cb')

            if self._serial_lidar is not None:
                try:
                    self._serial_lidar.resume()
                except Exception as e:
                    _log(f'serial lidar resume failed: {e}')

            with self._lock:
                self.state = 'idle'
                self.error = None
            _log('idle')
        except Exception as e:
            with self._lock:
                self.state = 'error'
                self.error = str(e)
            _log(f'stop failed: {e}')

    def _clear_shm(self):
        for p in glob.glob('/dev/shm/fastrtps_*') + glob.glob('/dev/shm/sem.fastrtps_*'):
            try:
                os.unlink(p)
            except OSError:
                pass

    def _reset_rplidar(self):
        try:
            import serial
        except ImportError:
            _log('pyserial missing, skipping lidar reset')
            return
        try:
            s = serial.Serial('/dev/ttyUSB0', 115200, timeout=2.0)
            s.dtr = False
            s.rts = False
            s.write(b'\xa5\x25')  # STOP
            time.sleep(0.3)
            s.write(b'\xa5\x40')  # RESET
            time.sleep(0.5)
            s.close()
        except Exception as e:
            _log(f'rplidar reset failed: {e}')

    def _ros_spin(self):
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import (QoSProfile, ReliabilityPolicy,
                                   DurabilityPolicy, HistoryPolicy)
            from sensor_msgs.msg import LaserScan, Imu
            from nav_msgs.msg import OccupancyGrid, Odometry
            from geometry_msgs.msg import PoseWithCovarianceStamped
            import tf2_ros
        except Exception as e:
            with self._lock:
                self.state = 'error'
                self.error = f'rclpy import failed: {e}'
            _log(self.error)
            return

        if not rclpy.ok():
            rclpy.init()
        node = Node('dashboard_slam_bridge')

        scan_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                              history=HistoryPolicy.KEEP_LAST, depth=5)
        map_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)

        def on_scan(msg):
            # Vectorized — was a Python loop over ~720 rays at 5.5 Hz on every
            # /scan publish. zip(...) over numpy arrays builds the final
            # (deg, mm) list at C speed.
            try:
                import numpy as _np
                ranges = _np.asarray(msg.ranges, dtype=_np.float32)
                n = ranges.size
                if n == 0:
                    pts = []
                else:
                    angles = msg.angle_min + _np.arange(n, dtype=_np.float32) * msg.angle_increment
                    valid = (ranges > msg.range_min) & (ranges < msg.range_max) & _np.isfinite(ranges)
                    deg = _np.degrees(angles[valid])
                    deg = _np.where(deg < 0, deg + 360.0, deg)
                    deg = _np.where(deg >= 360.0, deg - 360.0, deg)
                    mm = ranges[valid] * 1000.0
                    pts = list(zip(deg.tolist(), mm.tolist()))
            except Exception:
                pts = []
                a = msg.angle_min
                inc = msg.angle_increment
                rmin = msg.range_min
                rmax = msg.range_max
                for r in msg.ranges:
                    if rmin < r < rmax:
                        deg = math.degrees(a)
                        if deg < 0: deg += 360.0
                        elif deg >= 360.0: deg -= 360.0
                        pts.append((deg, r * 1000.0))
                    a += inc
            now = time.monotonic()
            with self._lock:
                self._scan_points = pts
                self._scan_ts = now
                self._scan_count += 1
                if now - self._scan_window_start >= 1.0:
                    self._scan_rate_hz = self._scan_count / (now - self._scan_window_start)
                    self._scan_count = 0
                    self._scan_window_start = now

        def on_map(msg):
            now = time.monotonic()
            # Coverage stats: vectorized so a 400×400 grid is two C-level sweeps
            # instead of 160 000 Python iterations per /map publish (1 Hz).
            try:
                import numpy as _np
                a = _np.asarray(msg.data, dtype=_np.int8)
                known = int((a != -1).sum())
                occupied = int((a > 50).sum())
            except Exception:
                known = 0; occupied = 0
                for v in msg.data:
                    if v != -1:
                        known += 1
                        if v > 50:
                            occupied += 1
            with self._lock:
                self._map = msg
                self._map_ts = now
                self._map_known_cells = known
                self._map_occupied_cells = occupied

        node.create_subscription(LaserScan, SCAN_TOPIC, on_scan, scan_qos)
        node.create_subscription(OccupancyGrid, MAP_TOPIC, on_map, map_qos)

        # /initialpose publisher — global_localize uses this to tell
        # slam_toolbox the bot's actual pose after scan-vs-map search.
        # TRANSIENT_LOCAL so a late-joining slam_toolbox subscriber still
        # picks up the message.
        initialpose_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                     durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                     history=HistoryPolicy.KEEP_LAST, depth=1)
        self._initialpose_pub = node.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', initialpose_qos)

        # /cmd_vel publishing was removed — cross-process discovery from this
        # process to imu_node was unreliable on this hardware. The dashboard
        # now integrates odometry locally and publishes TF directly.

        # Subscribe to /imu for yaw rate. imu_node no longer publishes /odom
        # or TF — we own those now, so we can integrate using in-process
        # driver state (no cross-process /cmd_vel needed).
        imu_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=10)
        self._imu_count = 0
        self._last_imu_omega_z = 0.0
        self._last_imu_ts = 0.0
        def on_imu(msg):
            self._imu_count += 1
            with self._lock:
                self._last_imu_omega_z = float(msg.angular_velocity.z)
                self._last_imu_ts = time.monotonic()
        node.create_subscription(Imu, '/imu', on_imu, imu_qos)

        # EKF fused twist (lidar_odom vx + IMU yaw-rate) — the measured
        # velocity used by _predict_motion and the confidence heuristic.
        def on_odom_filtered(msg):
            with self._lock:
                self._last_ekf_vx = float(msg.twist.twist.linear.x)
                self._last_ekf_vx_ts = time.monotonic()
        node.create_subscription(Odometry, '/odometry/filtered',
                                 on_odom_filtered, imu_qos)

        # Odometry + the odom→base_link TF are now owned by the EKF
        # (robot_localization) running in the SLAM launch. This process is a
        # pure consumer: it looks up TF for display, never publishes it.
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer, node)  # noqa: F841

        last_pose_lookup = 0.0
        try:
            while not self._ros_stop.is_set():
                rclpy.spin_once(node, timeout_sec=0.05)
                now = time.monotonic()
                # Mirror the live commanded velocity for the dashboard readout
                # (the EKF's cmd_vel_bridge consumes the same value via HTTP).
                if self._driver is not None:
                    try:
                        _, throttle, _ = self._driver.snapshot()
                        vx = (float(throttle) * float(self._driver.get_vel_factor())
                              * float(self._driver.get_motor_power()))
                        with self._lock:
                            self._last_cmd_vx = vx
                    except Exception:
                        pass
                if now - last_pose_lookup > 0.1:
                    last_pose_lookup = now
                    try:
                        t = tf_buffer.lookup_transform(
                            'map', 'base_link', rclpy.time.Time())
                        x = t.transform.translation.x
                        y = t.transform.translation.y
                        q = t.transform.rotation
                        theta = math.atan2(2 * (q.w * q.z + q.x * q.y),
                                           1 - 2 * (q.y * q.y + q.z * q.z))
                        # New policy: TF (EKF + cmd_vel + map→odom) is NOT
                        # the source of truth for self._pose. The bot icon
                        # and LiDAR-scan overlay should ONLY move when
                        # there's scan-vs-map evidence that the bot moved
                        # — i.e., a local_scan_match success. cmd_vel is
                        # uncalibrated and would otherwise drive the icon
                        # forward on every joystick press even when the
                        # bot didn't physically move that much, making the
                        # LiDAR dots appear to drift off the walls.
                        # We DO seed self._pose from TF exactly once at
                        # startup so the initial pose isn't None; after
                        # that, only scan-match updates it. TF carries the
                        # BASE yaw; self._pose stores LASER yaw (the custom
                        # matcher's convention) — convert at this boundary.
                        with self._lock:
                            need_seed = self._pose is None
                            if need_seed:
                                self._pose = (x, y, theta - LASER_NOSE_RAD)
                                self._pose_ts = now
                    except Exception:
                        pass
                    # Look up odom→base_link (published by the EKF) for the
                    # dashboard odom readout — the fused RF2O+IMU+cmd_vel pose.
                    try:
                        t = tf_buffer.lookup_transform(
                            'odom', 'base_link', rclpy.time.Time())
                        q = t.transform.rotation
                        oyaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                                          1 - 2 * (q.y * q.y + q.z * q.z))
                        with self._lock:
                            self._odom_xy = (t.transform.translation.x,
                                             t.transform.translation.y)
                            self._odom_yaw = oyaw
                            self._odom_ts = now
                    except Exception:
                        pass
        finally:
            self._initialpose_pub = None
            try:
                node.destroy_node()
            except Exception:
                pass
            _log('ros spin exited')
