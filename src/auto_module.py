"""Autonomous nav: A* on the SLAM OccupancyGrid + pure-pursuit drive.

Workflow:
  1. SLAM must be active (localization or mapping mode) so that we have a pose
     in the `map` frame and a current OccupancyGrid.
  2. User picks a goal (x,y) in map coords. set_goal() runs A* immediately and
     stashes the plan in self.path.
  3. go() spawns a control thread that runs pure-pursuit at CTRL_HZ until the
     bot is within GOAL_TOLERANCE_M of the goal, the user calls stop(), or
     something fails (SLAM lost, no progress, etc.).

The planner doesn't try to be clever — 8-connected A* with obstacle inflation
plus a line-of-sight smoothing pass. Good enough for the small indoor maps
this bot sees, and easy to reason about when it goes wrong.
"""
import heapq
import json
import math
import os
import sys
import threading
import time


# --- Tuning ---------------------------------------------------------------
# Planner
INFLATE_RADIUS_M = 0.20      # robot half-footprint + buffer; walls this thick
# Arrival tolerance. This is set by the PRECISION FLOOR of the hardware, not
# by ambition: the ESC won't move the bot below a high throttle, and lidar
# odometry doesn't count the first ~0.1-0.2 m of any move, so the smallest
# reliable hop is ~0.2 m and scan-match position itself wobbles ~0.1-0.2 m.
# Asking for tighter than ~0.25 m just makes the bot thrash forever around
# the goal. 0.25 m is honest for this platform.
GOAL_TOLERANCE_M = 0.25

# Control loop
CTRL_HZ = 20

# Per-hop limits. The controller drives in short straight "hops" and
# re-localizes (scan-vs-map) at every stop. Keeping each hop short bounds
# how far dead-reckoning has to carry between scan-matches — that is what
# makes the slow (~1 Hz) absolute-pose source usable: we only READ the pose
# while STOPPED, never while moving. (The previous controller read the
# ~1 Hz / frozen pose at 15 Hz while driving full-throttle, which is why it
# steered against stale data and veered off in random directions.)
# Coarse-to-fine hop sizing: each hop covers a FRACTION of the remaining
# distance, then we stop and re-localize. Far from the goal that fraction is a
# big move (few stops); as the goal nears, the same fraction yields smaller and
# smaller moves (more stops, finer precision) — exactly the behavior we want.
# We never drive the whole distance open-loop because odom can't measure
# distance accurately; fractional hops converge regardless of that error, and a
# real overshoot just flips the next hop to reverse.
APPROACH_FRACTION = 0.7
MAX_HOP_M = 1.50             # cap on a single (coarse, far-away) hop
MIN_HOP_M = 0.15             # smallest hop worth commanding (ESC bite floor)
# Re-localize PRIOR is TIME-based dead reckoning, NOT odom. In a long hallway
# the LiDAR scan is translationally aliased — sliding along the corridor barely
# changes the scan — so BOTH scan-match AND lidar_odom are blind to
# along-corridor distance and report almost no motion even when the bot drove
# meters. (An odom-derived prior therefore lands short of the true position and
# the search snaps the pose backward — the "frozen pose / veer / stuck" failure
# observed in the hallway.) Instead we trust what we KNOW: we commanded the bot
# to drive `hop` metres along a heading the IMU tracks reliably, so we predict
# it `hop` metres along that heading and let scan-match correct heading + the
# lateral offset (which the walls DO constrain) while leaving the along-corridor
# estimate near the dead-reckoned value.
NOMINAL_SPEED_MPS = 0.5      # ground speed at full throttle; converts hop<->time
# Run-1 evidence: hops under ~0.5 s are mostly ESC spin-up and move the bot
# 0-15 cm — inside scan-match noise, so near-goal iterations made no progress
# and tripped the watchdog. 0.5 s minimum guarantees real motion; a slight
# overshoot is fine (the next iteration backs in).
MIN_HOP_TIME_S = 0.50
MAX_HOP_TIME_S = 3.0         # 3 s ~= 1.5 m at full speed (coarse far-away hop)
# Re-localize search radius grows with the last hop so a big coarse move still
# re-acquires even if the dead-reckoned distance is a bit off (speed error).
SEARCH_RADIUS_BASE_M = 0.50
SEARCH_RADIUS_PER_HOP = 0.45
SEARCH_RADIUS_MAX_M = 1.20

# Throttle. This ESC+battery needs a high duty to move at all; the dashboard
# Driver scales these by speed_cap, so keep them at full and let speed_cap
# (default ~0.8) set the real speed.
DRIVE_THROTTLE = 1.0
ROTATE_THROTTLE = 1.0        # forward-arc throttle while turning to a heading
BACKUP_THROTTLE = 0.6        # magnitude for clearance nudges (sign set per use)

# Steering
STEER_MAG = 0.85             # steer magnitude during a rotate arc
HEADING_HOLD_KP = 1.4        # P gain holding a straight line during a hop
HEADING_HOLD_MAX = 0.45      # cap on corrective steer during a straight hop
HEADING_HOLD_DEADBAND_RAD = math.radians(3.0)

# One-time steer-sign calibration, run ONCE at the start of every drive before
# any hop. A brief probe arc + IMU yaw tells us which way set_cmd steers, so
# the straight-line heading-hold corrects in the RIGHT direction. Without this,
# a goal that starts dead-ahead skips the rotate phase (and thus the rotate's
# own sign check), leaving the sign at its default — if that default is wrong,
# the heading-hold AMPLIFIES drift and the bot curves off its line.
STEER_PROBE_MAG = 0.7
STEER_PROBE_THROTTLE = 1.0
STEER_PROBE_MAX_S = 1.2       # probe until a clear yaw signal OR this long
STEER_PROBE_TARGET_YAW_RAD = math.radians(12.0)  # stop probing once this clear
STEER_PROBE_MIN_YAW_RAD = math.radians(6.0)      # below this, too weak to trust
STEER_PROBE_CLEAR_M = 0.65    # room needed in the probe direction

# Rotate-to-heading. The steer sign is set ONCE up front by
# _calibrate_steer_sign (IMU probe); _rotate_to_heading does NOT re-probe/flip
# it — that misfired whenever a wall-backup interrupted the arc and corrupted
# the sign mid-turn.
ROTATE_TOL_RAD = math.radians(10.0)   # heading considered "aligned"
ROTATE_TIMEOUT_S = 12.0

# Forward vs reverse: if reaching the next waypoint needs a heading change
# greater than this, the goal is "behind" the nose.
# 90° is the natural split — beyond it, reverse needs less turning.
FORWARD_REVERSE_DECISION_RAD = math.radians(90.0)

# When the goal is BEHIND us: if it's closer than this, just back straight
# into it; if farther, turn around to FACE it and drive forward. Raised from
# 0.45 m: run 2 spent ~45 s fighting a turn-around in a corridor (0.19 m of
# side clearance) for a goal 1.4 m straight behind — backing in is a single
# smooth maneuver, the rear obstacle cone guards it, and heading-hold works
# in reverse. Turn-arounds are now reserved for genuinely long routes.
REVERSE_MAX_M = 2.50

# Close-range endgame. Within this distance of the GOAL we stop provoking big
# maneuvers: aim straight at the goal, never turn around, and SKIP the full-lock
# pre-rotate — just take small hops and let the gentle heading-hold (capped
# steer, while moving) curve in. This avoids the "overshoot the goal -> goal is
# now behind -> swing a 1 m full-lock arc to realign" veer that wrecks the
# approach right next to the target. A slight overshoot becomes a small reverse
# nudge instead. Heading at the goal doesn't matter — only position does.
CLOSE_RANGE_M = 0.70

# Turn-around style is chosen by the space the LiDAR sees around the bot:
#  * straight 180° sweep — fast but needs a big clear area (the bot traces a
#    half-circle of ~2× its turning radius); require this much clearance both
#    AHEAD and to the turn side.
#  * 3-point (K) turn — fits a tight spot by shuffling forward/reverse in
#    place; only needs a bit of room ahead AND behind.
FULL_TURN_CLEAR_M = 1.00          # ahead + turn-side clearance to allow a 180
THREE_POINT_MIN_CLEAR_M = 0.40    # min room ahead/behind to shuffle a 3-pt turn
# Legs are now HEADING-bounded, not just time-bounded: each leg drives until
# it has rotated THREE_POINT_LEG_GAIN_RAD (or hits a wall / the time cap).
# Run 2 measured ~10-15°/leg with 0.7 s legs (ESC spin-up ate most of each),
# so 8 legs only managed 80° of a 130° turn and gave up — twice.
THREE_POINT_SEG_S = 1.40          # time cap per leg (safety, not the target)
THREE_POINT_LEG_GAIN_RAD = math.radians(30.0)   # end leg once rotated this much
THREE_POINT_STALL_GAIN_RAD = math.radians(6.0)  # 2 legs under this = wedged
THREE_POINT_MAX_SEGS = 12         # give up after this many forward/reverse legs
TURN_CLEAR_CONE_DEG = 30.0        # cone half-width for the clearance probes

# Escape recovery: when a maneuver is wedged (can't rotate/turn — walls block
# the arc), instead of hard-erroring, drive toward the MORE-OPEN direction
# (front vs rear, by LIVE LiDAR — which is reliable regardless of localization)
# to regain room, then re-localize and retry. Bounded so it can't wander while
# lost: a short move, and only a few consecutive tries before we give up.
ESCAPE_TIME_S = 1.2               # ~0.6 m toward the opening per attempt
MAX_ESCAPES = 4                   # consecutive escapes (reset on any real progress)

# LiDAR mount offset. The RPLidar A1 is mounted rotated ~90° on this car, so
# the robot's physical FORWARD (the +x base_link "nose") sits at NOSE_DEG in
# the LASER frame. SLAM's pose theta is the LASER heading, so the nose points
# at (theta + NOSE_OFFSET) in the map frame, and a forward drive translates
# along that nose axis — NOT along theta. This is the SAME +92° axis
# lidar_odom.py projects velocity onto (its NOSE_DEG). Everything the
# controller reasons about "which way am I pointing to move toward a map
# point" must use the nose heading, and obstacle cones must be centered on
# the nose (laser angle NOSE_DEG forward, NOSE_DEG+180 for reverse) — not on
# laser-0°, which points out the bot's right side.
NOSE_DEG = 92.0
NOSE_OFFSET_RAD = math.radians(NOSE_DEG)

# Obstacle stop during a hop (cone centered on the travel direction).
OBSTACLE_CONE_DEG = 35.0
DRIVE_OBSTACLE_M = 0.35
WALL_DETECT_M = 0.35         # front-arc wall guard during a rotate

# ESC brake-priming (a short pulse + neutral so the ESC engages the requested
# direction instead of reading the first reversed pulse as a brake).
BACKUP_BRAKE_PULSE_S = 0.30
BACKUP_RESET_NEUTRAL_S = 0.30
BACKUP_DRIVE_S = 0.70

# Stop-and-settle before every scan-match so the LiDAR sees a static scene.
SETTLE_S = 0.45
RELOCALIZE_MAX_TRIES = 3
# Only re-anchor (scan-match) after a rotate if it swung far — a big arc
# translates the bot up to ~0.5 m so the anchor needs refreshing, but a small
# alignment arc barely moves it and the IMU heading is trustworthy; skipping
# the confirm match saves ~2 s on most iterations.
ROTATE_CONFIRM_MIN_SWING_RAD = math.radians(60.0)
# If progress stalls but we're already within this factor of the arrival
# tolerance, declare arrived instead of erroring ("pretty much there" should
# not be a failure — run 1 errored 31 cm from the goal on a noisy match).
# Judged on the BEST distance achieved (the long-run test got to 0.31 m,
# wiggled out to 0.43 m, and the current-distance check missed by 3 cm).
ARRIVED_LOOSE_FACTOR = 1.6
ARRIVED_LOOSE_MAX_CURRENT = 2.4   # ...but only if currently within this factor
# Aim-point lookahead: never steer at a path vertex closer than this. The A*
# path can put a vertex centimetres away (start-cell rounding); bearing to a
# 10 cm-away point is hypersensitive and meaningless for a car — the long-run
# test did a full-lock 45° rotate to "reach" a vertex 10 cm away, flinging
# itself 0.5 m sideways, then rotated 54° back. Skip such vertices.
LOOKAHEAD_MIN_M = 0.45
# Close-range alignment: if the goal sits far off BOTH chassis axes (beside
# the bot), gentle nudges just orbit it — the heading-hold can't swing 50-80°
# in a 0.5 s hop. Pivot roughly in place with K-turn legs first, then nudge.
CLOSE_ALIGN_TOL_RAD = math.radians(30.0)
MAX_CLOSE_ALIGNS = 2              # consecutive pivots before just nudging anyway
# Calibrated steer sign persists across runs (probe still runs each go()).
STEER_SIGN_FILE = '/home/deepracer/.auto_steer_sign'

# Whole-run guards.
MAX_ITERS = 50
LOST_ABORT_CONFIDENCE = 0.28
PROGRESS_WINDOW_ITERS = 4    # over this many hops, goal-distance must drop
PROGRESS_MIN_M = 0.08        # by at least this much, or we declare stuck


# --- Run logging ------------------------------------------------------------
# Every auto-drive run writes a JSONL event log to RUN_LOG_DIR (one JSON object
# per line, flushed immediately so a crash loses nothing). Includes a 10 Hz
# telemetry trace, every motor-command transition, and every decision with its
# inputs — enough to reconstruct a run offline and diagnose wrong-direction
# turns / inefficiency without being there. `latest.jsonl` symlinks the
# current/most-recent run.
RUN_LOG_DIR = '/home/deepracer/autodrive_logs'


def _round_for_json(v):
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, (list, tuple)):
        return [_round_for_json(x) for x in v]
    if isinstance(v, dict):
        return {k: _round_for_json(x) for k, x in v.items()}
    return v


class RunLogger:
    """Append-only JSONL event log, one file per drive run."""

    def __init__(self):
        self._f = None
        self._wlock = threading.Lock()
        self._t0 = 0.0
        self.path = None

    def start(self, meta):
        self.end('superseded')   # safety: close any leftover open run
        try:
            os.makedirs(RUN_LOG_DIR, exist_ok=True)
            name = time.strftime('run_%Y%m%d_%H%M%S') + '.jsonl'
            path = os.path.join(RUN_LOG_DIR, name)
            with self._wlock:
                self._f = open(path, 'a', buffering=1)
                self._t0 = time.monotonic()
                self.path = path
            link = os.path.join(RUN_LOG_DIR, 'latest.jsonl')
            try:
                if os.path.islink(link) or os.path.exists(link):
                    os.unlink(link)
                os.symlink(path, link)
            except OSError:
                pass
            self.log('run_start', **meta)
            _log(f'run log: {path}')
        except Exception as e:
            print(f'[auto] run log start failed: {e}', file=sys.stderr, flush=True)

    def log(self, ev, **fields):
        with self._wlock:
            f = self._f
            if f is None:
                return
            try:
                rec = {'t': round(time.monotonic() - self._t0, 3),
                       'clock': round(time.time(), 3), 'ev': ev}
                rec.update(_round_for_json(fields))
                f.write(json.dumps(rec, default=str) + '\n')
            except Exception:
                pass

    def end(self, result, **fields):
        if self._f is None:
            return
        self.log('run_end', result=result, **fields)
        with self._wlock:
            try:
                self._f.close()
            except Exception:
                pass
            self._f = None


RUNLOG = RunLogger()


def _log(msg):
    # Timestamped so dashboard.log lines can be merged with the JSONL trace.
    ts = time.time()
    stamp = time.strftime('%H:%M:%S', time.localtime(ts)) + f'.{int(ts % 1 * 1000):03d}'
    print(f'[auto {stamp}] {msg}', file=sys.stderr, flush=True)
    RUNLOG.log('log', msg=msg)


def _wrap_angle(a):
    while a > math.pi:  a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


def _line_clear(p1, p2, blocked, ox, oy, res):
    """True if the straight line p1->p2 (world coords) is free of `blocked` cells."""
    x0, y0 = p1
    x1, y1 = p2
    dist = math.hypot(x1 - x0, y1 - y0)
    n = max(2, int(dist / (res * 0.5)) + 1)
    h, w = blocked.shape
    for i in range(n + 1):
        t = i / n
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        cx = int((x - ox) / res)
        cy = int((y - oy) / res)
        if not (0 <= cy < h and 0 <= cx < w):
            return False
        if blocked[cy, cx]:
            return False
    return True


class AutoNav:
    STATES = ('idle', 'planning', 'ready', 'driving', 'arrived', 'aborted', 'error')

    def __init__(self, slam, driver):
        self.slam = slam
        self.driver = driver
        self._lock = threading.Lock()
        self.state = 'idle'
        self.error = None
        self.goal = None           # (x, y) in map frame
        self.path = []             # list of (x, y) world waypoints
        self._stop_evt = threading.Event()
        self._thread = None
        self._dist_remaining = None
        self._progress_start_ts = 0.0
        self._progress_start_dist = 0.0
        # Steer sign: +1 = set_cmd(+) steers toward increasing yaw (CCW) when
        # driving forward, -1 = inverted. _rotate_to_heading self-calibrates
        # this from the IMU on the first turn (flips it if the heading error
        # isn't shrinking), so a wrong default corrects itself within ~1 s.
        # Can also be pinned via POST /auto/steer_polarity {"polarity": -1}.
        # Default = last calibrated value (persisted), so even a weak/skipped
        # probe starts from the sign that worked last run, not a guess.
        self._steer_polarity = 1
        try:
            with open(STEER_SIGN_FILE) as f:
                self._steer_polarity = -1 if int(f.read().strip()) < 0 else 1
        except (OSError, ValueError):
            pass
        # Dead-reckoning anchor: the last STOPPED scan-match pose in the map
        # frame, paired with the EKF odom (x, y, yaw) at that instant. Between
        # stops we never read the slow absolute pose — we compose the live
        # odom delta onto this anchor instead. Reset at the start of each run.
        self._anchor_pose = None     # (x, y, theta) map frame
        self._anchor_odom = None     # ((x, y), yaw) odom frame
        # Time-based dead-reckoned map-frame displacement since the anchor (set
        # after each commanded move, consumed as the re-localize prior). This —
        # not odom — is how we predict position, because odom is blind to
        # along-corridor motion. (dx, dy) in metres.
        self._predicted_disp = (0.0, 0.0)
        self._last_hop_m = 0.0       # size of the most recent commanded move
        self._escape_count = 0       # consecutive wedge-escape recoveries
        self._last_cmd = (0.0, 0.0)  # last (steer, throttle) sent (for logging)
        self._telem_stop = threading.Event()  # stops the 10 Hz run trace

    # ---------- public API -------------------------------------------------

    def status(self):
        with self._lock:
            return {
                'state': self.state,
                'error': self.error,
                'goal': ({'x': self.goal[0], 'y': self.goal[1]}
                         if self.goal is not None else None),
                # Path is intentionally returned even for 'arrived' / 'aborted'
                # so the UI keeps showing the last attempted route until the
                # user picks a new goal.
                'path': [{'x': p[0], 'y': p[1]} for p in self.path],
                'dist_remaining_m': (round(self._dist_remaining, 3)
                                     if self._dist_remaining is not None else None),
            }

    def set_goal(self, gx, gy):
        if self.slam is None:
            return False, 'SLAM unavailable'
        if self.slam.state != 'active':
            return False, f'SLAM not active (state={self.slam.state})'
        pose = self._read_slam_pose()
        if pose is None:
            return False, 'no pose yet — wait for SLAM to localize'
        grid = self._read_slam_map()
        if grid is None:
            return False, 'no map yet'
        # Cancel any in-progress drive before re-planning.
        self._abort_current()
        with self._lock:
            self.state = 'planning'
            self.goal = (float(gx), float(gy))
            self.path = []
            self.error = None
            self._dist_remaining = None
        try:
            path = self._plan(pose, (float(gx), float(gy)), grid)
        except Exception as e:
            _log(f'plan failed: {e}')
            with self._lock:
                self.state = 'error'
                self.error = f'plan error: {e}'
            return False, str(e)
        if path is None:
            with self._lock:
                self.state = 'error'
                self.error = 'no path found (goal unreachable or surrounded by obstacles)'
            return False, self.error
        plen = self._path_length(path)
        with self._lock:
            self.path = path
            self.state = 'ready'
            self._dist_remaining = plen
        _log(f'goal set ({gx:.2f}, {gy:.2f}) from pose ({pose[0]:.2f}, '
             f'{pose[1]:.2f}, {math.degrees(pose[2]):.0f}deg): '
             f'{len(path)} waypoints, {plen:.2f} m')
        return True, f'{len(path)} waypoints, {plen:.2f} m'

    def go(self):
        with self._lock:
            if self.state != 'ready':
                return False, f'cannot go from {self.state} (need ready)'
            if not self.path:
                return False, 'no path planned'
            self.state = 'driving'
            self.error = None
            self._stop_evt.clear()
            self._progress_start_ts = time.monotonic()
            self._progress_start_dist = self._dist_remaining or 0.0
        self._thread = threading.Thread(target=self._drive_loop, daemon=True)
        self._thread.start()
        return True, 'driving'

    def stop(self):
        RUNLOG.log('user_stop')
        self._abort_current()
        with self._lock:
            # Only flip to aborted if we were actually driving — otherwise leave
            # the state alone so the UI keeps the last plan visible.
            if self.state == 'driving':
                self.state = 'aborted'
        try:
            self.driver.stop()
        except Exception:
            pass
        return True, 'stopped'

    def clear(self):
        """Reset to idle, dropping any plan / goal."""
        self._abort_current()
        with self._lock:
            self.state = 'idle'
            self.error = None
            self.goal = None
            self.path = []
            self._dist_remaining = None
        return True, 'cleared'

    def set_steer_polarity(self, p):
        """Set the steer polarity used by auto-drive. p=+1 standard,
        p=-1 inverts (use when SLAM yaw and physical yaw disagree)."""
        with self._lock:
            self._steer_polarity = -1 if float(p) < 0 else 1
        _log(f'steer polarity set to {self._steer_polarity:+d}')
        return self._steer_polarity

    def get_steer_polarity(self):
        with self._lock:
            return self._steer_polarity

    # ---------- internals --------------------------------------------------

    def _cmd(self, angle, throttle):
        """Single funnel for motor commands: logs every (steer, throttle)
        TRANSITION to the run log (the 20 Hz loops repeat the same command, so
        only changes are recorded), then forwards to the driver."""
        key = (round(float(angle), 3), round(float(throttle), 3))
        if key != self._last_cmd:
            self._last_cmd = key
            RUNLOG.log('cmd', steer=key[0], throttle=key[1])
        self.driver.set_cmd(angle, throttle)

    def _telemetry_snapshot(self):
        """One row of everything we can observe, for the 10 Hz trace."""
        with self.slam._lock:
            spose = self.slam._pose
            conf = self.slam._confidence
            vx = getattr(self.slam, '_last_ekf_vx', None)
            omega = getattr(self.slam, '_last_imu_omega_z', None)
            scan_age = (time.monotonic() - self.slam._scan_ts
                        if self.slam._scan_ts else None)
        xy, oyaw, _ = self._read_odom()
        dr = self._deadreckon()
        front = self._front_min_distance_mm(OBSTACLE_CONE_DEG, NOSE_DEG)
        rear = self._front_min_distance_mm(OBSTACLE_CONE_DEG, NOSE_DEG + 180.0)
        return {
            'dr': dr, 'slam_pose': spose,
            'odom_xy': xy, 'odom_yaw': oyaw,
            'ekf_vx': vx, 'omega_z': omega, 'conf': conf,
            'scan_age_s': scan_age,
            'front_mm': front, 'rear_mm': rear,
            'cmd': self._last_cmd,
        }

    def _telemetry_loop(self):
        while not self._telem_stop.is_set():
            try:
                RUNLOG.log('telem', **self._telemetry_snapshot())
            except Exception:
                pass
            if self._telem_stop.wait(0.1):
                return

    def _read_slam_pose(self):
        """Snapshot the current pose under the slam manager's lock."""
        with self.slam._lock:
            p = self.slam._pose
        return p

    def _read_slam_map(self):
        with self.slam._lock:
            m = self.slam._map
        return m

    def _read_slam_confidence(self):
        with self.slam._lock:
            return self.slam._confidence

    def _abort_current(self):
        self._stop_evt.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._thread = None
        try:
            self.driver.stop()
        except Exception:
            pass

    @staticmethod
    def _path_length(path):
        d = 0.0
        for i in range(1, len(path)):
            d += math.hypot(path[i][0] - path[i-1][0],
                            path[i][1] - path[i-1][1])
        return d

    def _plan(self, start_pose, goal_xy, grid):
        """A* over the OccupancyGrid; returns list[(x,y)] or None."""
        import numpy as np
        info = grid.info
        w, h = int(info.width), int(info.height)
        res = float(info.resolution)
        ox = float(info.origin.position.x)
        oy = float(info.origin.position.y)
        if w <= 0 or h <= 0:
            raise ValueError('empty map')

        # OccupancyGrid.data is signed int8: -1 unknown, 0 free, 100 occupied.
        # Ros2 messages on this stack come as a python `array.array` of signed
        # bytes — convert robustly.
        raw = bytes(bytearray((b & 0xFF) for b in grid.data))
        arr = np.frombuffer(raw, dtype=np.int8).reshape((h, w))
        blocked = (arr > 50)
        unknown = (arr == -1)

        # Inflate obstacles by INFLATE_RADIUS_M so the planner gives walls a
        # comfortable margin (otherwise the bot can clip a wall while turning).
        r_cells = max(1, int(round(INFLATE_RADIUS_M / res)))
        try:
            import cv2
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * r_cells + 1, 2 * r_cells + 1))
            cost_blocked = cv2.dilate(blocked.astype('uint8'), k) > 0
        except ImportError:
            cost_blocked = blocked

        def w2c(x, y):
            return int((x - ox) / res), int((y - oy) / res)

        sx, sy, _ = start_pose
        gx, gy = goal_xy
        sc = w2c(sx, sy)
        gc = w2c(gx, gy)
        if not (0 <= sc[0] < w and 0 <= sc[1] < h):
            raise ValueError('start outside map')
        if not (0 <= gc[0] < w and 0 <= gc[1] < h):
            raise ValueError('goal outside map')
        if cost_blocked[gc[1], gc[0]]:
            # Try a small spiral to snap to the nearest free cell — common case:
            # user clicks right on a wall pixel.
            snap = self._nearest_free(cost_blocked, gc, max_radius=int(0.5 / res))
            if snap is None:
                raise ValueError('goal is inside an inflated obstacle')
            gc = snap

        D = math.sqrt(2.0)
        nbrs = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                (1, 1, D), (1, -1, D), (-1, 1, D), (-1, -1, D))

        def heur(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        open_set = [(heur(sc, gc), 0.0, sc)]
        g_score = {sc: 0.0}
        came = {sc: None}
        found = False
        while open_set:
            _, g, cur = heapq.heappop(open_set)
            if cur == gc:
                found = True
                break
            if g > g_score.get(cur, float('inf')):
                continue
            cx, cy = cur
            # Are we currently INSIDE the inflated zone? This happens when the
            # bot starts hugging a wall (within INFLATE_RADIUS_M of it). When so,
            # we must let the search climb OUT (step to blocked OR free cells),
            # otherwise A* can't expand at all and reports "no path" even though
            # a route obviously exists. From a free cell the normal rule holds:
            # never enter inflation.
            cur_blocked = cost_blocked[cy, cx]
            for dx, dy, step in nbrs:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                nbr_blocked = cost_blocked[ny, nx]
                if nbr_blocked and not cur_blocked:
                    continue   # don't drive a free path into a wall margin
                # Diagonal corner-cut guard (only meaningful in free space).
                if dx != 0 and dy != 0 and not cur_blocked:
                    if cost_blocked[cy, nx] or cost_blocked[ny, cx]:
                        continue
                cost = step
                if nbr_blocked:
                    cost *= 5.0  # heavily penalize staying in the wall margin
                elif unknown[ny, nx]:
                    cost *= 3.0  # discourage planning through unknown
                tg = g + cost
                npos = (nx, ny)
                if tg < g_score.get(npos, float('inf')):
                    g_score[npos] = tg
                    came[npos] = cur
                    heapq.heappush(open_set, (tg + heur(npos, gc), tg, npos))

        if not found:
            return None

        # Reconstruct cell-by-cell, then convert to world centers.
        cells = []
        cur = gc
        while cur is not None:
            cells.append(cur)
            cur = came[cur]
        cells.reverse()
        path = [((cc + 0.5) * res + ox, (cr + 0.5) * res + oy)
                for (cc, cr) in cells]
        # Smooth with greedy line-of-sight -> the minimal set of corner
        # vertices. We intentionally do NOT chunk into many small waypoints:
        # the segmented controller drives straight toward the next corner and
        # re-plans after every hop, so corners are all it needs.
        return self._smooth_path(path, cost_blocked, ox, oy, res)

    @staticmethod
    def _nearest_free(blocked, cell, max_radius):
        """BFS outward from `cell` looking for an unblocked cell."""
        h, w = blocked.shape
        cx, cy = cell
        for r in range(1, max_radius + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and not blocked[ny, nx]:
                        return (nx, ny)
        return None

    @staticmethod
    def _smooth_path(path, blocked, ox, oy, res):
        if len(path) < 3:
            return path
        out = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if _line_clear(path[i], path[j], blocked, ox, oy, res):
                    break
                j -= 1
            out.append(path[j])
            i = j
        return out


    # ---------- odometry / dead-reckoning ---------------------------------

    def _read_odom(self):
        """Snapshot the EKF odom->base_link pose (the fast, motion-tracking
        source: IMU yaw + lidar_odom velocity). Returns ((x,y)|None, yaw, ts)."""
        with self.slam._lock:
            xy = self.slam._odom_xy
            yaw = self.slam._odom_yaw
            ts = self.slam._odom_ts
        return xy, yaw, ts

    def _deadreckon(self):
        """Current pose estimate in the MAP frame, composed from the last
        stopped scan-match (the anchor) plus the EKF odom delta since then.

        This is what we read WHILE MOVING — never self._pose, which only
        updates ~1 Hz (or not at all in mapping mode). The anchor pins the
        absolute map position; odom (IMU yaw + lidar velocity) supplies the
        smooth, high-rate delta. Falls back to the raw SLAM pose before the
        first anchor exists."""
        if self._anchor_pose is None or self._anchor_odom is None:
            return self._read_slam_pose()
        xy, oyaw, _ = self._read_odom()
        if xy is None:
            return self._anchor_pose
        ax, ay, ath = self._anchor_pose
        (ox0, oy0), oyaw0 = self._anchor_odom
        # NOTE: the xy here is only a rough fallback; the real position prior is
        # time-based dead reckoning in _relocalize. What this is used for is the
        # YAW (IMU/EKF), which is reliable. xy uses the raw (unscaled) odom delta.
        dox, doy = xy[0] - ox0, xy[1] - oy0
        # Rotate the odom-frame delta into the map frame by the anchor's
        # map<-odom yaw offset, then add to the anchor position.
        phi = ath - oyaw0
        cp, sp = math.cos(phi), math.sin(phi)
        px = ax + cp * dox - sp * doy
        py = ay + sp * dox + cp * doy
        pth = _wrap_angle(ath + _wrap_angle(oyaw - oyaw0))
        return (px, py, pth)

    def _yaw_offset_rad(self):
        """The user's manual yaw-offset (slam_module slider), radians."""
        try:
            return math.radians(self.slam.get_yaw_offset_deg())
        except Exception:
            return 0.0

    def _nose_heading(self, theta):
        """Physical NOSE heading in the TRUE map frame.

        Pose/matcher theta lives in the yaw-offset-shifted frame (the manual
        display slider), while bearings to map points are true-frame. Mixing
        them biased every heading decision by exactly the offset — measured
        in the long-run log as travel direction = commanded heading − offset.
        Subtract the offset, then add the 92° lidar-mount nose rotation."""
        return _wrap_angle(theta - self._yaw_offset_rad() + NOSE_OFFSET_RAD)

    def _live_heading(self):
        """Physical NOSE heading in the map frame — the direction a forward
        drive actually moves the bot."""
        dr = self._deadreckon()
        return self._nose_heading(dr[2]) if dr is not None else None

    def _relocalize(self):
        """STOP-time scan-vs-map match. Sets a fresh dead-reckon anchor.

        Centers the bounded local search on the odom-predicted pose so a hop
        that exceeded the raw search radius is still re-acquired. On success
        slam_module also updates its own self._pose (so the dashboard icon
        tracks). Returns the map pose (x,y,theta) or None."""
        prev_anchor = self._anchor_pose
        # Prior = TIME-based dead reckoning: the anchor advanced by the
        # displacement we commanded since it (self._predicted_disp, map frame),
        # with the heading the IMU has been tracking. This reaches the bot's
        # true position even in an aliased corridor where odom/scan-match see
        # almost no motion — so the search locks on instead of snapping back.
        predicted = None
        if prev_anchor is not None:
            th = self._deadreckon()[2]   # IMU/EKF-tracked laser heading
            dx, dy = self._predicted_disp
            predicted = (prev_anchor[0] + dx, prev_anchor[1] + dy, th)
        # Search radius grows with the last move so a coarse hop re-acquires
        # even if the dead-reckoned distance is off (speed error); tight + fast
        # for the fine approach near the goal.
        radius = min(SEARCH_RADIUS_MAX_M,
                     SEARCH_RADIUS_BASE_M + SEARCH_RADIUS_PER_HOP * self._last_hop_m)
        step = 0.10 if radius <= SEARCH_RADIUS_BASE_M + 1e-6 else 0.15
        t_start = time.monotonic()
        first_predicted = predicted
        pose = None
        attempts = []
        fallback = False
        for _ in range(RELOCALIZE_MAX_TRIES):
            if self._stop_evt.is_set():
                return None
            msg = None
            try:
                p, msg = self.slam.local_scan_match(
                    search_radius_m=radius, step_m=step, prior_pose=predicted)
            except Exception as e:
                _log(f'relocalize error: {e}')
                p = None
                msg = f'exception: {e}'
            attempts.append({'prior': predicted, 'msg': msg})
            if p is not None:
                pose = p
                break
            # Retry with the SAME dead-reckoned prior. The old code nulled the
            # prior here, which made the search re-center on slam._pose — and
            # in run 2 that pose was background-localize garbage (120° off),
            # which we then adopted as our anchor and spiraled. The IMU prior
            # is the trustworthy reference; never abandon it.
            if self._stop_evt.wait(0.35):
                return None
        if pose is None:
            # No match at all: trust the dead-reckoned prediction (IMU heading
            # + commanded displacement) over the module's pose for the same
            # reason. Next iteration's match re-anchors.
            pose = predicted if predicted is not None else self._read_slam_pose()
            fallback = True
        if pose is None:
            RUNLOG.log('relocalize', ok=False, radius=radius, step=step,
                       prior=first_predicted, attempts=attempts,
                       predicted_disp=self._predicted_disp,
                       last_hop_m=self._last_hop_m)
            return None
        # How far the match moved us off the dead-reckoned prediction — large
        # jumps mean either the prior or the match is lying.
        jump = None
        if first_predicted is not None:
            jump = {'d_m': math.hypot(pose[0] - first_predicted[0],
                                      pose[1] - first_predicted[1]),
                    'dyaw_deg': math.degrees(
                        _wrap_angle(pose[2] - first_predicted[2]))}
        prev_jump = None
        if prev_anchor is not None:
            prev_jump = {'d_m': math.hypot(pose[0] - prev_anchor[0],
                                           pose[1] - prev_anchor[1]),
                         'dyaw_deg': math.degrees(
                             _wrap_angle(pose[2] - prev_anchor[2]))}
        RUNLOG.log('relocalize', ok=True, pose=pose, radius=radius, step=step,
                   prior=first_predicted, attempts=attempts,
                   fallback_to_slam_pose=fallback,
                   predicted_disp=self._predicted_disp,
                   last_hop_m=self._last_hop_m,
                   jump_from_prior=jump, moved_since_prev_anchor=prev_jump,
                   took_s=time.monotonic() - t_start)
        xy, oyaw, _ = self._read_odom()
        self._anchor_pose = pose
        self._anchor_odom = ((xy[0], xy[1]), oyaw) if xy is not None else None
        self._predicted_disp = (0.0, 0.0)   # consumed
        return pose

    # ---------- drive loop -------------------------------------------------

    def _drive_loop(self):
        """Segmented stop-and-go drive.

        Each iteration: STOP -> scan-match (authoritative pose) -> re-plan ->
        decide forward/reverse -> rotate to the segment heading (IMU-tracked
        arc) -> STOP -> confirm heading -> drive ONE bounded straight hop.
        Repeat until verified at the goal.

        Absolute pose is read ONLY while stopped; everything during motion
        rides on IMU yaw + odom (the signals that are reliable in motion)."""
        try:
            self._anchor_pose = None
            self._anchor_odom = None
            self._last_hop_m = 0.0     # first relocalize uses the base radius
            self._predicted_disp = (0.0, 0.0)
            self._escape_count = 0
            dist_hist = []
            close_aligns = 0   # consecutive close-range pivot-in-place turns

            # AutoNav is the SOLE localizer while driving. The background
            # continuous-localize loop can't track our full-lock arcs (1 Hz,
            # ±25° search) and was committing corridor-symmetric garbage poses
            # mid-maneuver while ALSO holding the match lock against our own
            # relocalize — the run-2 "turning in circles" feedback loop.
            try:
                self.slam.set_continuous_paused(True)
            except AttributeError:
                pass

            RUNLOG.start({
                'goal': self.goal,
                'path': self.path,
                'start_pose': self._read_slam_pose(),
                'slam_mode': getattr(self.slam, 'mode', None),
                'steer_polarity': self._steer_polarity,
                'yaw_offset_deg': (self.slam.get_yaw_offset_deg()
                                   if hasattr(self.slam, 'get_yaw_offset_deg')
                                   else None),
                'constants': {
                    'APPROACH_FRACTION': APPROACH_FRACTION,
                    'MAX_HOP_M': MAX_HOP_M, 'MIN_HOP_M': MIN_HOP_M,
                    'MIN_HOP_TIME_S': MIN_HOP_TIME_S,
                    'MAX_HOP_TIME_S': MAX_HOP_TIME_S,
                    'NOMINAL_SPEED_MPS': NOMINAL_SPEED_MPS,
                    'STEER_MAG': STEER_MAG,
                    'HEADING_HOLD_KP': HEADING_HOLD_KP,
                    'HEADING_HOLD_MAX': HEADING_HOLD_MAX,
                    'ROTATE_TOL_RAD': ROTATE_TOL_RAD,
                    'CLOSE_RANGE_M': CLOSE_RANGE_M,
                    'REVERSE_MAX_M': REVERSE_MAX_M,
                    'GOAL_TOLERANCE_M': GOAL_TOLERANCE_M,
                    'NOSE_DEG': NOSE_DEG,
                    'DRIVE_OBSTACLE_M': DRIVE_OBSTACLE_M,
                    'WALL_DETECT_M': WALL_DETECT_M,
                },
            })
            self._telem_stop.clear()
            threading.Thread(target=self._telemetry_loop, daemon=True).start()

            # Calibrate the steer sign ONCE up front so straight-line hops
            # (which skip the rotate phase when the goal starts dead ahead)
            # hold heading in the correct direction instead of curving away.
            self._calibrate_steer_sign()
            if self._stop_evt.is_set():
                return

            for it in range(1, MAX_ITERS + 1):
                if self._stop_evt.is_set():
                    return

                # 1. STOP + settle + localize (the only trusted absolute pose).
                self._cmd(0.0, 0.0)
                if self._stop_evt.wait(SETTLE_S):
                    return
                pose = self._relocalize()
                if pose is None:
                    self._fail('could not localize (scan-vs-map failed)')
                    return
                conf = self._read_slam_confidence()
                if conf < LOST_ABORT_CONFIDENCE:
                    self._fail(f'SLAM confidence low ({conf:.2f})')
                    return

                gx, gy = self.goal
                dist_goal = math.hypot(gx - pose[0], gy - pose[1])
                with self._lock:
                    self._dist_remaining = dist_goal
                nose = self._nose_heading(pose[2])
                _log(f'iter {it}: at ({pose[0]:.2f}, {pose[1]:.2f}), '
                     f'laser {math.degrees(pose[2]):.0f}deg / nose '
                     f'{math.degrees(nose):.0f}deg, {dist_goal:.2f} m to goal')
                RUNLOG.log('iter', it=it, pose=pose,
                           nose_deg=math.degrees(nose),
                           dist_goal=dist_goal, conf=conf,
                           escape_count=self._escape_count)

                # 2. Arrived? This pose IS a settled scan-match (we stopped +
                # waited SETTLE_S before localizing), so it's the verification
                # — no separate second match. The OLD code ran a second
                # scan-match here whose result it then ignored, re-planning
                # from the stale pre-match pose: that's why an overshoot got
                # detected but the bot still drove forward instead of backing
                # up. Now every iteration plans from the one authoritative pose,
                # so an overshoot (goal now behind the nose) reverses next.
                if dist_goal < GOAL_TOLERANCE_M:
                    _log(f'arrived ({dist_goal*100:.0f} cm from goal)')
                    with self._lock:
                        self.state = 'arrived'
                    return

                # 3. Progress watchdog on the BEST distance so far — a single
                # noisy backward scan-match can't fail the run by inflating
                # the current distance (run 1 died exactly that way: one bad
                # match jumped 0.4 m back at 0.31 m from the goal). And if we
                # stall but are already nearly there, that's an arrival, not
                # an error.
                best_so_far = min(dist_hist[-1], dist_goal) if dist_hist else dist_goal
                dist_hist.append(best_so_far)
                if len(dist_hist) > PROGRESS_WINDOW_ITERS:
                    gained = dist_hist[-PROGRESS_WINDOW_ITERS - 1] - best_so_far
                    RUNLOG.log('watchdog', it=it, gained_m=gained,
                               best_m=best_so_far, dist_m=dist_goal,
                               dist_hist=dist_hist[-PROGRESS_WINDOW_ITERS - 1:])
                    if gained < PROGRESS_MIN_M:
                        # Judge "close enough" by the BEST distance achieved:
                        # near the goal the bot wiggles (scan noise + nudge
                        # overshoot), and the long-run test stalled at 0.43 m
                        # current after having BEEN at 0.31 m — that's an
                        # arrival, not an error.
                        if (best_so_far < GOAL_TOLERANCE_M * ARRIVED_LOOSE_FACTOR
                                and dist_goal < GOAL_TOLERANCE_M * ARRIVED_LOOSE_MAX_CURRENT):
                            _log(f'progress stalled {dist_goal*100:.0f} cm from '
                                 f'goal (best {best_so_far*100:.0f} cm) — close '
                                 'enough, arriving')
                            RUNLOG.log('arrived_loose', dist_m=dist_goal,
                                       best_m=best_so_far)
                            with self._lock:
                                self.state = 'arrived'
                            return
                        self._fail(
                            f'no progress ({gained*100:.0f} cm over '
                            f'{PROGRESS_WINDOW_ITERS} hops) — stuck')
                        return

                # 4. Re-plan from the just-localized pose.
                grid = self._read_slam_map()
                if grid is None:
                    self._fail('lost map')
                    return
                try:
                    path = self._plan(pose, self.goal, grid)
                except Exception as e:
                    self._fail(f'plan error: {e}')
                    return
                if not path or len(path) < 2:
                    if dist_goal < GOAL_TOLERANCE_M * 1.5:
                        with self._lock:
                            self.state = 'arrived'
                        return
                    self._fail('no path to goal')
                    return
                with self._lock:
                    self.path = path
                # Aim at the first vertex at least LOOKAHEAD_MIN_M away — a
                # nearer vertex gives a hypersensitive, meaningless bearing
                # (the source of the long-run test's pointless 45° rotates).
                # The path is LOS-smoothed, so skipping a near vertex cuts at
                # most a sub-half-metre corner inside the inflated margin.
                aim = path[-1]
                for v in path[1:]:
                    if math.hypot(v[0] - pose[0], v[1] - pose[1]) >= LOOKAHEAD_MIN_M:
                        aim = tuple(v)
                        break
                RUNLOG.log('plan', it=it, path=path, aim=aim,
                           length_m=self._path_length(path))

                # 5. Decide the maneuver. err_fwd = how far the NOSE must swing
                # to face the aim point (nose = where the bot actually drives).
                # Close to the goal, aim STRAIGHT AT THE GOAL (ignore detour
                # vertices) and enter the gentle endgame below.
                close = dist_goal < CLOSE_RANGE_M
                if close:
                    aim = self.goal
                bearing = math.atan2(aim[1] - pose[1], aim[0] - pose[0])
                nose = self._nose_heading(pose[2])
                err_fwd = _wrap_angle(bearing - nose)
                dist_to_aim = math.hypot(aim[0] - pose[0], aim[1] - pose[1])

                # Goal well BEHIND and not close enough to just back into it:
                # turn around to FACE it (straight-180 or 3-point per space),
                # then re-localize and drive forward next iteration. NOT in
                # close range — there a slight overshoot must NOT trigger a big
                # turn; it's handled by a small reverse nudge instead.
                if (not close
                        and abs(err_fwd) > FORWARD_REVERSE_DECISION_RAD
                        and dist_to_aim > REVERSE_MAX_M):
                    _log(f'iter {it}: goal behind ({math.degrees(err_fwd):+.0f}deg, '
                         f'{dist_to_aim:.2f} m) — turning around to face it')
                    RUNLOG.log('decide', it=it, kind='turn_around', aim=aim,
                               bearing_deg=math.degrees(bearing),
                               nose_deg=math.degrees(nose),
                               err_fwd_deg=math.degrees(err_fwd),
                               dist_to_aim=dist_to_aim, close=close)
                    self._last_hop_m = MAX_HOP_M  # a turn can translate a lot
                    if self._turn_around(bearing):
                        # A turn deliberately repositions us (often farther from
                        # the goal) — restart the progress window so the watchdog
                        # doesn't count the maneuver itself as "no progress".
                        dist_hist = []
                        continue
                    # Turn-around couldn't complete. If the user/SLAM stopped us
                    # or a hard error fired, bail; otherwise we're just too
                    # boxed-in to turn — fall back to backing straight in.
                    if self._stop_evt.is_set() or self.state == 'error':
                        return
                    # Too boxed-in to turn — re-localize (the turn legs moved
                    # us) and back straight in toward the goal instead.
                    _log('iter: cannot turn around here — backing in instead')
                    pose = self._relocalize()
                    if pose is None:
                        self._fail('lost pose after turn-around attempt')
                        return
                    bearing = math.atan2(aim[1] - pose[1], aim[0] - pose[0])
                    direction, target_heading = -1, _wrap_angle(bearing + math.pi)
                    RUNLOG.log('decide', it=it, kind='back_in_fallback',
                               aim=aim, bearing_deg=math.degrees(bearing),
                               target_heading_deg=math.degrees(target_heading))
                # Otherwise face it forward, or (behind + close) back straight in.
                elif abs(err_fwd) <= FORWARD_REVERSE_DECISION_RAD:
                    direction, target_heading = 1, bearing
                else:
                    direction, target_heading = -1, _wrap_angle(bearing + math.pi)
                RUNLOG.log('decide', it=it,
                           kind='forward' if direction > 0 else 'reverse',
                           aim=aim, close=close,
                           bearing_deg=math.degrees(bearing),
                           nose_deg=math.degrees(nose),
                           err_fwd_deg=math.degrees(err_fwd),
                           dist_to_aim=dist_to_aim,
                           target_heading_deg=math.degrees(target_heading))

                # 6. Rotate (arc) to the target heading if not aligned, then
                # CONFIRM with a fresh scan-match. Errors are vs the NOSE
                # heading (where the bot points), not raw laser theta.
                # In CLOSE range we skip this full-lock pre-rotate entirely — the
                # gentle heading-hold during the short hop does the (small)
                # steering, so a near-goal misalignment can't fling a 1 m arc.
                # (recompute nose — pose may have been refreshed in the fallback)
                nose = self._nose_heading(pose[2])
                swing = _wrap_angle(target_heading - nose)
                arc_extra_m = 0.0
                # Close-range with the goal far off the chassis axis (e.g. an
                # overshoot left it beside us): nudges along the current axis
                # would orbit the goal (iters 7-10 of the long-run test slid
                # 0.31→0.43 m AROUND it). Pivot roughly in place with K-turn
                # legs — bounded, wall-aware, ~0.2 m of translation — so the
                # next nudge actually points at the goal. Full-lock arcs stay
                # forbidden here (they fling ~0.5 m past a near goal).
                if close:
                    if abs(swing) <= CLOSE_ALIGN_TOL_RAD:
                        close_aligns = 0
                    elif close_aligns < MAX_CLOSE_ALIGNS:
                        close_aligns += 1
                        _log(f'iter {it}: goal {math.degrees(swing):+.0f}deg off '
                             f'the {"nose" if direction > 0 else "tail"} at close '
                             'range — pivoting in place')
                        RUNLOG.log('close_align', it=it, n=close_aligns,
                                   swing_deg=math.degrees(swing),
                                   direction=direction,
                                   target_heading_deg=math.degrees(target_heading))
                        ok = self._three_point_turn(target_heading,
                                                    1 if swing > 0 else -1)
                        if self._stop_evt.is_set() or self.state == 'error':
                            return
                        self._cmd(0.0, 0.0)
                        if self._stop_evt.wait(SETTLE_S):
                            return
                        self._predicted_disp = (0.0, 0.0)
                        self._last_hop_m = 0.5
                        dist_hist = []   # deliberate repositioning
                        RUNLOG.log('close_align_done', it=it, ok=ok)
                        continue
                    # else: pivots exhausted — nudge anyway; the loose-arrive
                    # check ends the run once progress genuinely stalls.
                if not close and abs(swing) > ROTATE_TOL_RAD:
                    _log(f'iter {it}: rotate to {math.degrees(target_heading):.0f}deg, '
                         f'then drive {"FWD" if direction > 0 else "REV"}')
                    if not self._rotate_to_heading(target_heading):
                        # Stopped or hard error -> bail. Otherwise we're wedged
                        # (couldn't arc past the walls) — back out toward open
                        # space and retry from a fresh pose instead of erroring.
                        if self._stop_evt.is_set() or self.state == 'error':
                            return
                        if not self._escape_to_open():
                            return
                        dist_hist = []   # recovery repositioned us — fresh window
                        continue
                    self._cmd(0.0, 0.0)
                    if self._stop_evt.wait(SETTLE_S):
                        return
                    if abs(swing) > ROTATE_CONFIRM_MIN_SWING_RAD:
                        # Big swing — the arc translated the bot substantially;
                        # re-anchor and confirm before committing to a hop.
                        self._last_hop_m = 0.5   # the arc translates ~this far
                        self._predicted_disp = (0.0, 0.0)  # translation unknown
                        pose = self._relocalize()
                        if pose is None:
                            self._fail('lost pose after rotate')
                            return
                        direction, target_heading = self._decide_direction(pose, aim)
                        nose_after = self._nose_heading(pose[2])
                        confirm_err = _wrap_angle(target_heading - nose_after)
                        RUNLOG.log('rotate_confirm', it=it, pose=pose,
                                   nose_after_deg=math.degrees(nose_after),
                                   target_heading_deg=math.degrees(target_heading),
                                   err_deg=math.degrees(confirm_err),
                                   direction=direction,
                                   ok=abs(confirm_err) <= ROTATE_TOL_RAD * 1.8)
                        if abs(confirm_err) > ROTATE_TOL_RAD * 1.8:
                            _log('iter: heading not confirmed — re-rotating next iter')
                            continue
                    else:
                        # Small alignment arc: the IMU heading is trustworthy
                        # and the translation is small — skip the confirm
                        # scan-match (saves ~2 s) and widen the next search
                        # to absorb the arc's unmeasured translation.
                        arc_extra_m = 0.4
                        self._predicted_disp = (0.0, 0.0)
                        RUNLOG.log('rotate_confirm_skipped', it=it,
                                   swing_deg=math.degrees(swing))

                # 7. Drive ONE closed-loop hop toward the corner. We cover only
                # a FRACTION of the remaining distance, then stop + re-localize
                # next iteration. This is deliberate: odom (lidar velocity)
                # under-reports — it can't tell us precisely how far we've gone —
                # so trying to nail the full distance in one open-loop shot
                # overshoots. Homing in by fractions converges geometrically
                # regardless of the odom scale error, and if we DO overshoot,
                # the goal lands behind the nose next iteration and we reverse.
                dist_to_aim = math.hypot(aim[0] - pose[0], aim[1] - pose[1])
                hop = min(MAX_HOP_M, max(MIN_HOP_M, APPROACH_FRACTION * dist_to_aim))
                # Never command a hop longer than what's actually left (a tiny
                # residual shouldn't be inflated up to MIN_HOP_M and overshoot).
                hop = min(hop, dist_to_aim)
                _log(f'iter {it}: drive {"FWD" if direction > 0 else "REV"} '
                     f'{hop:.2f} m of {dist_to_aim:.2f} m remaining toward '
                     f'({aim[0]:.2f}, {aim[1]:.2f})')
                # arc_extra_m widens the next relocalize search when a
                # confirm-skipped rotate left unmeasured arc translation.
                self._last_hop_m = hop + arc_extra_m
                res = self._drive_hop(direction, hop, target_heading)
                RUNLOG.log('hop_result', it=it, result=res, hop_m=hop,
                           direction=direction,
                           target_heading_deg=math.degrees(target_heading))
                if res == 'abort':
                    return
                if res == 'obstacle':
                    # Clear in the opposite direction, then re-plan next iter.
                    # Net motion is uncertain — let the next relocalize search
                    # without a dead-reckon offset.
                    self._predicted_disp = (0.0, 0.0)
                    self._nudge(-direction)
                    continue
                # Made real progress this iteration — reset the consecutive
                # escape counter so a later tight spot gets fresh recovery tries.
                self._escape_count = 0
                # Completed the hop: dead-reckon that we moved `hop` metres along
                # the driven heading (forward) or opposite it (reverse). This is
                # the prior the next relocalize will trust — see _relocalize.
                self._predicted_disp = (
                    hop * direction * math.cos(target_heading),
                    hop * direction * math.sin(target_heading))

            self._fail(f'did not reach goal in {MAX_ITERS} hops')
        except Exception as e:
            _log(f'drive_loop exception: {e}')
            import traceback
            RUNLOG.log('exception', error=str(e), tb=traceback.format_exc())
            self._fail(f'drive error: {e}')
        finally:
            try:
                self.slam.set_continuous_paused(False)
            except AttributeError:
                pass
            self._telem_stop.set()
            with self._lock:
                final_state = self.state
                final_err = self.error
                final_dist = self._dist_remaining
            RUNLOG.end(final_state, error=final_err,
                       dist_remaining_m=final_dist,
                       final_pose=self._read_slam_pose())

    def _calibrate_steer_sign(self):
        """Determine the steer sign from the IMU before any hop runs.

        Commands a brief arc and checks which way the yaw actually goes. Sets
        self._steer_polarity so the heading-hold (and rotate) steer the correct
        way. Probes forward if the nose is clear, else reverse (which inverts
        the steer->yaw relation, so the deduced sign is flipped to match)."""
        _, yaw0, _ = self._read_odom()
        if yaw0 is None:
            _log('steer cal: no odom yaw yet — keeping default sign')
            return
        fwd = True
        d_front = self._front_min_distance_mm(OBSTACLE_CONE_DEG, NOSE_DEG)
        if d_front is not None and d_front < STEER_PROBE_CLEAR_M * 1000.0:
            d_rear = self._front_min_distance_mm(OBSTACLE_CONE_DEG, NOSE_DEG + 180.0)
            if d_rear is None or d_rear > STEER_PROBE_CLEAR_M * 1000.0:
                fwd = False
            else:
                _log('steer cal: boxed front+rear — skipping probe, keeping '
                     f'sign {self._steer_polarity:+d}')
                return
        RUNLOG.log('steer_cal_start', fwd=fwd, yaw0=yaw0,
                   front_mm=d_front,
                   rear_mm=self._front_min_distance_mm(
                       OBSTACLE_CONE_DEG, NOSE_DEG + 180.0))
        if not fwd and not self._prime(-1):
            return
        probe_thr = STEER_PROBE_THROTTLE if fwd else -STEER_PROBE_THROTTLE
        center = NOSE_DEG if fwd else (NOSE_DEG + 180.0)
        tick = 1.0 / CTRL_HZ
        t0 = time.monotonic()
        dyaw = 0.0
        # Arc until the yaw has changed by a CLEARLY readable amount (not the
        # near-noise few degrees a too-short probe gave before), or time/space
        # runs out. We trust the accumulated sign only if it's decisive.
        while time.monotonic() - t0 < STEER_PROBE_MAX_S:
            if self._stop_evt.is_set():
                self._cmd(0.0, 0.0)
                return
            _, yawn, _ = self._read_odom()
            if yawn is not None:
                dyaw = _wrap_angle(yawn - yaw0)
            if abs(dyaw) >= STEER_PROBE_TARGET_YAW_RAD:
                break
            d = self._front_min_distance_mm(OBSTACLE_CONE_DEG, center)
            if d is not None and d < WALL_DETECT_M * 1000.0:
                break  # ran out of room; judge with whatever we have
            self._cmd(STEER_PROBE_MAG, probe_thr)
            if self._stop_evt.wait(tick):
                self._cmd(0.0, 0.0)
                return
        self._cmd(0.0, 0.0)
        if self._stop_evt.wait(0.25):       # settle so the final reading is clean
            return
        _, yaw1, _ = self._read_odom()
        if yaw1 is not None:
            dyaw = _wrap_angle(yaw1 - yaw0)
        if abs(dyaw) < STEER_PROBE_MIN_YAW_RAD:
            _log(f'steer cal: weak yaw response ({math.degrees(dyaw):+.0f}deg) '
                 f'— keeping sign {self._steer_polarity:+d}')
            RUNLOG.log('steer_cal_result', fwd=fwd,
                       dyaw_deg=math.degrees(dyaw), decisive=False,
                       sign=self._steer_polarity,
                       took_s=time.monotonic() - t0)
            return
        # +steer forward -> +yaw means sign +1; a reverse probe inverts that.
        if fwd:
            self._steer_polarity = 1 if dyaw > 0 else -1
        else:
            self._steer_polarity = -1 if dyaw > 0 else 1
        try:
            with open(STEER_SIGN_FILE, 'w') as f:
                f.write(f'{self._steer_polarity}\n')
        except OSError:
            pass
        _log(f'steer cal: {"fwd" if fwd else "rev"} probe +steer -> '
             f'{math.degrees(dyaw):+.0f}deg yaw, sign {self._steer_polarity:+d}')
        RUNLOG.log('steer_cal_result', fwd=fwd, dyaw_deg=math.degrees(dyaw),
                   decisive=True, sign=self._steer_polarity,
                   took_s=time.monotonic() - t0)

    def _decide_direction(self, pose, aim):
        """Pick forward vs reverse for the next segment and return
        (direction, target_nose_heading).

        Compares the bearing to the aim point against the NOSE heading
        (pose theta + nose offset — the way the bot actually drives), NOT the
        raw laser theta. Forward if the nose is within
        FORWARD_REVERSE_DECISION_RAD of the bearing; otherwise reverse, with
        the nose facing AWAY from the aim point so the rear leads into it.
        target_nose_heading is where we want the NOSE to point in either case."""
        bearing = math.atan2(aim[1] - pose[1], aim[0] - pose[0])
        nose = self._nose_heading(pose[2])
        err_fwd = _wrap_angle(bearing - nose)
        if abs(err_fwd) <= FORWARD_REVERSE_DECISION_RAD:
            return 1, bearing
        return -1, _wrap_angle(bearing + math.pi)

    def _rotate_to_heading(self, target_heading, arc_dir=1):
        """Swing the chassis to `target_heading` by arcing.

        Ackermann can't pivot, so a heading change is an arc. Crucially we arc in
        whichever direction has ROOM: a forward arc and a reverse arc (with the
        same steer formula) rotate the bot the SAME way, so when the nose is
        against a wall we flip to reversing — which keeps turning while backing
        into the open instead of bashing the wall. Heading is tracked from the
        IMU/EKF yaw (reliable in motion); the steer SIGN comes from the up-front
        `_calibrate_steer_sign` (we do NOT re-probe/flip it here — when the bot
        was wall-blocked that check misfired and corrupted the sign mid-turn).
        Returns True when aligned, False on stop / wedged (caller may escape)."""
        arc_dir = 1 if arc_dir >= 0 else -1
        tick = 1.0 / CTRL_HZ
        t0 = time.monotonic()
        switches = 0
        MAX_SWITCHES = 6   # bounded shuffle if both directions are tight
        start_heading = self._live_heading()
        RUNLOG.log('rotate_start',
                   target_deg=math.degrees(target_heading),
                   heading_deg=(math.degrees(start_heading)
                                if start_heading is not None else None),
                   arc_dir=arc_dir, polarity=self._steer_polarity)
        last_dbg = 0.0

        while time.monotonic() - t0 < ROTATE_TIMEOUT_S:
            if self._stop_evt.is_set():
                self._cmd(0.0, 0.0)
                return False
            th = self._live_heading()
            if th is None:
                self._fail('lost heading during rotate')
                return False
            err = _wrap_angle(target_heading - th)
            now = time.monotonic()
            if now - last_dbg >= 0.5:
                last_dbg = now
                RUNLOG.log('rotate_progress', heading_deg=math.degrees(th),
                           err_deg=math.degrees(err), arc_dir=arc_dir,
                           elapsed_s=now - t0)
            if abs(err) < ROTATE_TOL_RAD:
                self._cmd(0.0, 0.0)
                _log(f'rotate: aligned (err {math.degrees(err):+.0f}deg)')
                RUNLOG.log('rotate_done', ok=True,
                           err_deg=math.degrees(err), switches=switches,
                           elapsed_s=time.monotonic() - t0)
                return True

            # Is the way we're arcing blocked? (cone centered on the travel
            # direction: nose for a forward arc, rear for a reverse arc.)
            center = NOSE_DEG if arc_dir > 0 else (NOSE_DEG + 180.0)
            d = self._front_min_distance_mm(OBSTACLE_CONE_DEG, center)
            if d is not None and d < WALL_DETECT_M * 1000.0:
                # Blocked this way — check the OTHER direction (which turns the
                # same way) and switch to it if it's clear.
                other_center = (NOSE_DEG + 180.0) if arc_dir > 0 else NOSE_DEG
                d_other = self._front_min_distance_mm(OBSTACLE_CONE_DEG, other_center)
                if d_other is not None and d_other < WALL_DETECT_M * 1000.0:
                    _log(f'rotate: blocked both ways (fwd/rev {d:.0f}/'
                         f'{d_other:.0f} mm) — wedged')
                    RUNLOG.log('rotate_done', ok=False, reason='wedged',
                               d_mm=d, d_other_mm=d_other, switches=switches,
                               elapsed_s=time.monotonic() - t0)
                    return False   # caller escapes
                if switches >= MAX_SWITCHES:
                    _log('rotate: too many direction switches — wedged')
                    RUNLOG.log('rotate_done', ok=False,
                               reason='too_many_switches', switches=switches,
                               elapsed_s=time.monotonic() - t0)
                    return False
                self._cmd(0.0, 0.0)
                arc_dir = -arc_dir
                switches += 1
                if arc_dir < 0 and not self._prime(-1):   # engage reverse
                    return False
                _log(f'rotate: arcing {"reverse" if arc_dir < 0 else "forward"} '
                     f'(other side blocked at {d:.0f} mm)')
                RUNLOG.log('rotate_switch', arc_dir=arc_dir, blocked_mm=d,
                           switches=switches)
                continue

            steer = self._steer_polarity * arc_dir * math.copysign(STEER_MAG, err)
            self._cmd(steer, arc_dir * ROTATE_THROTTLE)
            if self._stop_evt.wait(tick):
                self._cmd(0.0, 0.0)
                return False

        _log(f'rotate: timeout aligning to {math.degrees(target_heading):.0f}deg')
        th = self._live_heading()
        RUNLOG.log('rotate_done', ok=False, reason='timeout',
                   heading_deg=(math.degrees(th) if th is not None else None),
                   target_deg=math.degrees(target_heading), switches=switches,
                   elapsed_s=time.monotonic() - t0)
        return False   # caller may try an escape recovery

    def _turn_around(self, target_heading):
        """Swing the NOSE to face `target_heading` (a ~U-turn for a behind goal).

        Chooses the style from the LiDAR clearance around the bot: a straight
        continuous arc if there's room to sweep the half-circle, else a 3-point
        (K) turn that pivots in a tight spot. Returns True if it ends facing the
        target, False on stop/abort."""
        nose = self._live_heading()
        if nose is None:
            self._fail('lost heading before turn-around')
            return False
        err = _wrap_angle(target_heading - nose)
        turn_dir = 1 if err > 0 else -1   # +1 = CCW, -1 = CW (shortest way round)

        # Clearance ahead and to the side we'd sweep toward. A continuous 180
        # sweeps forward and out to that side; a 3-point turn does not.
        front = self._front_min_distance_mm(TURN_CLEAR_CONE_DEG, NOSE_DEG)
        side = self._front_min_distance_mm(
            TURN_CLEAR_CONE_DEG, NOSE_DEG + 90.0 * turn_dir)
        fm = -1.0 if front is None else front / 1000.0
        sm = -1.0 if side is None else side / 1000.0
        front_ok = front is None or front > FULL_TURN_CLEAR_M * 1000.0
        side_ok = side is None or side > FULL_TURN_CLEAR_M * 1000.0

        RUNLOG.log('turn_around', style='arc' if (front_ok and side_ok) else '3point',
                   turn_dir=turn_dir, front_m=fm, side_m=sm,
                   nose_deg=math.degrees(nose),
                   target_deg=math.degrees(target_heading),
                   err_deg=math.degrees(err))
        if front_ok and side_ok:
            _log(f'turn-around: open (front {fm:.2f} m, side {sm:.2f} m) '
                 f'— straight arc {"CCW" if turn_dir > 0 else "CW"}')
            return self._rotate_to_heading(target_heading)
        _log(f'turn-around: tight (front {fm:.2f} m, side {sm:.2f} m) '
             f'— 3-point turn {"CCW" if turn_dir > 0 else "CW"}')
        return self._three_point_turn(target_heading, turn_dir)

    def _three_point_turn(self, target_heading, turn_dir):
        """Pivot toward `target_heading` by shuffling forward/reverse in place.

        The K-turn trick: a forward leg with the wheels toward turn_dir and a
        reverse leg with the wheels the other way BOTH rotate the bot the same
        way (turn_dir), while keeping it roughly in one spot — so it fits where
        a continuous arc won't. The steer formula
        `steer_polarity * leg_dir * copysign(MAG, turn_dir)` already yields
        turn_dir rotation for either leg sign. Legs are short and watch the wall
        in their own travel direction; we alternate until facing the target."""
        tick = 1.0 / CTRL_HZ
        # Start with whichever direction has more room.
        f = self._front_min_distance_mm(TURN_CLEAR_CONE_DEG, NOSE_DEG)
        r = self._front_min_distance_mm(TURN_CLEAR_CONE_DEG, NOSE_DEG + 180.0)
        leg_dir = -1 if (r is not None and (f is None or r > f)) else 1
        leg_gains = []   # heading gained per leg — 2 stalled legs = wedged

        for leg in range(THREE_POINT_MAX_SEGS):
            if self._stop_evt.is_set():
                self._cmd(0.0, 0.0)
                return False
            nose = self._live_heading()
            if nose is None:
                self._fail('lost heading during 3-point turn')
                return False
            if abs(_wrap_angle(target_heading - nose)) < ROTATE_TOL_RAD:
                self._cmd(0.0, 0.0)
                _log('3-point: facing target')
                return True

            center = NOSE_DEG if leg_dir > 0 else (NOSE_DEG + 180.0)
            clr = self._front_min_distance_mm(TURN_CLEAR_CONE_DEG, center)
            if clr is not None and clr < THREE_POINT_MIN_CLEAR_M * 1000.0:
                # This way is blocked — flip and check the other side.
                leg_dir = -leg_dir
                center = NOSE_DEG if leg_dir > 0 else (NOSE_DEG + 180.0)
                clr = self._front_min_distance_mm(TURN_CLEAR_CONE_DEG, center)
                if clr is not None and clr < THREE_POINT_MIN_CLEAR_M * 1000.0:
                    _log('3-point: boxed in front and back — cannot turn')
                    return False   # caller falls back to backing in

            if leg_dir < 0 and not self._prime(-1):
                return False
            steer = self._steer_polarity * leg_dir * math.copysign(STEER_MAG, turn_dir)
            RUNLOG.log('three_point_leg', leg=leg, leg_dir=leg_dir,
                       steer=steer, clearance_mm=clr,
                       nose_deg=(math.degrees(nose) if nose is not None else None),
                       target_deg=math.degrees(target_heading))
            leg_start_nose = nose
            t0 = time.monotonic()
            while time.monotonic() - t0 < THREE_POINT_SEG_S:
                if self._stop_evt.is_set():
                    self._cmd(0.0, 0.0)
                    return False
                nose = self._live_heading()
                if nose is not None and abs(_wrap_angle(target_heading - nose)) < ROTATE_TOL_RAD:
                    self._cmd(0.0, 0.0)
                    _log('3-point: facing target')
                    return True
                # Heading-bounded leg: keep driving until this leg has rotated
                # its share (time cap is a backstop). Time-capped 0.7 s legs
                # only gained 10-15° each — ESC spin-up ate most of the leg.
                if (nose is not None and leg_start_nose is not None
                        and abs(_wrap_angle(nose - leg_start_nose))
                            >= THREE_POINT_LEG_GAIN_RAD):
                    break
                d = self._front_min_distance_mm(TURN_CLEAR_CONE_DEG, center)
                if d is not None and d < WALL_DETECT_M * 1000.0:
                    break   # ran into the wall this leg — end the leg, alternate
                self._cmd(steer, leg_dir * ROTATE_THROTTLE)
                if self._stop_evt.wait(tick):
                    self._cmd(0.0, 0.0)
                    return False
            self._cmd(0.0, 0.0)
            if self._stop_evt.wait(0.2):
                return False
            end_nose = self._live_heading()
            gain = (abs(_wrap_angle(end_nose - leg_start_nose))
                    if end_nose is not None and leg_start_nose is not None
                    else 0.0)
            leg_gains.append(gain)
            RUNLOG.log('three_point_leg_end', leg=leg,
                       gain_deg=math.degrees(gain),
                       nose_deg=(math.degrees(end_nose)
                                 if end_nose is not None else None))
            if (len(leg_gains) >= 2
                    and leg_gains[-1] < THREE_POINT_STALL_GAIN_RAD
                    and leg_gains[-2] < THREE_POINT_STALL_GAIN_RAD):
                _log('3-point: two stalled legs — wedged')
                return False   # caller falls back / escapes
            leg_dir = -leg_dir   # alternate forward/reverse

        nose = self._live_heading()
        if nose is not None and abs(_wrap_angle(target_heading - nose)) < ROTATE_TOL_RAD * 1.5:
            return True
        _log(f'3-point: incomplete after {THREE_POINT_MAX_SEGS} legs')
        return False   # caller falls back to backing in

    def _escape_to_open(self):
        """Wedge recovery: drive toward the more-open direction (front vs rear,
        by LIVE LiDAR) to regain room, then re-localize. Collision-safe because
        clearance sensing doesn't depend on knowing our map pose. Bounded in
        distance and in consecutive attempts. Returns True if recovered."""
        self._escape_count += 1
        if self._escape_count > MAX_ESCAPES:
            self._fail(f'wedged — out of escape attempts ({MAX_ESCAPES})')
            return False
        front = self._front_min_distance_mm(TURN_CLEAR_CONE_DEG, NOSE_DEG)
        rear = self._front_min_distance_mm(TURN_CLEAR_CONE_DEG, NOSE_DEG + 180.0)
        fm = float('inf') if front is None else front / 1000.0
        rm = float('inf') if rear is None else rear / 1000.0
        if max(fm, rm) < THREE_POINT_MIN_CLEAR_M:
            self._fail(f'escape: boxed on all sides (front {fm:.2f} m, '
                       f'rear {rm:.2f} m) — cannot recover')
            return False
        direction = -1 if rm >= fm else 1     # head toward the more-open side
        _log(f'escape #{self._escape_count}: wedged (front {fm:.2f} m, '
             f'rear {rm:.2f} m) — {"reversing" if direction < 0 else "forward"} '
             f'toward open space')
        RUNLOG.log('escape', n=self._escape_count, front_m=fm, rear_m=rm,
                   direction=direction)
        if direction < 0 and not self._prime(-1):
            return False
        center = NOSE_DEG if direction > 0 else (NOSE_DEG + 180.0)
        tick = 1.0 / CTRL_HZ
        t0 = time.monotonic()
        while time.monotonic() - t0 < ESCAPE_TIME_S:
            if self._stop_evt.is_set():
                self._cmd(0.0, 0.0)
                return False
            d = self._front_min_distance_mm(OBSTACLE_CONE_DEG, center)
            if d is not None and d < DRIVE_OBSTACLE_M * 1000.0:
                break   # reached the far wall on the open side — stop short
            self._cmd(0.0, direction * DRIVE_THROTTLE)
            if self._stop_evt.wait(tick):
                self._cmd(0.0, 0.0)
                return False
        self._cmd(0.0, 0.0)
        if self._stop_evt.wait(SETTLE_S):
            return False
        # Position is uncertain after the recovery move — clear the dead-reckon
        # offset and re-localize with a generous search.
        self._predicted_disp = (0.0, 0.0)
        self._last_hop_m = ESCAPE_TIME_S * NOMINAL_SPEED_MPS
        pose = self._relocalize()
        if pose is None:
            self._fail('escape: lost pose after recovery move')
            return False
        return True

    def _drive_hop(self, direction, hop_dist, target_heading):
        """Drive ONE short, TIME-BOUNDED impulse in `direction` (+1 forward,
        -1 reverse), holding `target_heading` with the IMU.

        We do NOT measure distance from odom. lidar-velocity odom under-reports
        badly and variably (observed ~2.5x), so an odom-distance hop drives far
        past the target AND outruns the scan-matcher — which then loses track
        and trips SLAM's lost-track safety, aborting the run. Instead each hop
        is a brief impulse whose DURATION is a rough function of the requested
        distance, capped short (MAX_HOP_TIME_S). The bot then STOPS and the next
        iteration's settled scan-match measures the true new position. The
        control loop closes on scan-match, not on in-motion odometry, so it
        converges regardless of the odom scale error and never outruns
        localization. Distance accuracy per hop doesn't matter — the feedback
        loop does.

        Returns 'ok' (impulse done), 'obstacle' (blocked), or 'abort'."""
        tick = 1.0 / CTRL_HZ
        if direction < 0 and not self._prime(-1):
            return 'abort'
        hop_time = max(MIN_HOP_TIME_S,
                       min(MAX_HOP_TIME_S, hop_dist / NOMINAL_SPEED_MPS))
        # Obstacle cone centered on the travel direction in the LASER frame:
        # nose for forward, nose+180 for reverse (_front_min_distance_mm wraps).
        center = NOSE_DEG if direction > 0 else (NOSE_DEG + 180.0)
        start_heading = self._live_heading()
        RUNLOG.log('hop_start', direction=direction, hop_m=hop_dist,
                   hop_time_s=hop_time,
                   target_heading_deg=math.degrees(target_heading),
                   heading_deg=(math.degrees(start_heading)
                                if start_heading is not None else None))
        t0 = time.monotonic()
        last_dbg = 0.0

        while time.monotonic() - t0 < hop_time:
            if self._stop_evt.is_set():
                self._cmd(0.0, 0.0)
                return 'abort'
            d = self._front_min_distance_mm(OBSTACLE_CONE_DEG, center)
            if d is not None and d < DRIVE_OBSTACLE_M * 1000.0:
                self._cmd(0.0, 0.0)
                _log(f'hop: obstacle {d:.0f} mm '
                     f'{"ahead" if direction > 0 else "behind"} — stop')
                RUNLOG.log('hop_obstacle', d_mm=d, direction=direction,
                           elapsed_s=time.monotonic() - t0)
                return 'obstacle'

            th = self._live_heading()
            if th is None:
                self._cmd(0.0, 0.0)
                self._fail('lost heading during hop')
                return 'abort'
            err = _wrap_angle(target_heading - th)
            if abs(err) < HEADING_HOLD_DEADBAND_RAD:
                steer = 0.0
            else:
                mag = min(HEADING_HOLD_MAX, HEADING_HOLD_KP * abs(err))
                steer = self._steer_polarity * direction * math.copysign(mag, err)
            now = time.monotonic()
            if now - last_dbg >= 0.4:
                last_dbg = now
                RUNLOG.log('hop_hold', heading_deg=math.degrees(th),
                           err_deg=math.degrees(err), steer=steer,
                           elapsed_s=now - t0)
            self._cmd(steer, direction * DRIVE_THROTTLE)
            if self._stop_evt.wait(tick):
                self._cmd(0.0, 0.0)
                return 'abort'
        self._cmd(0.0, 0.0)
        th = self._live_heading()
        RUNLOG.log('hop_end',
                   heading_deg=(math.degrees(th) if th is not None else None),
                   drift_deg=(math.degrees(_wrap_angle(th - target_heading))
                              if th is not None else None),
                   elapsed_s=time.monotonic() - t0)
        return 'ok'

    def _front_min_distance_mm(self, cone_deg=OBSTACLE_CONE_DEG, center_deg=0.0):
        """Min LiDAR range (mm) within a cone of half-width cone_deg centered
        on center_deg (0 = forward, 180 = rear). None if no readings.

        Uses the slam_module's cached scan (deg, mm tuples)."""
        with self.slam._lock:
            pts = list(self.slam._scan_points)
        best = None
        for a_deg, d_mm in pts:
            off = abs(((a_deg - center_deg + 180.0) % 360.0) - 180.0)
            if off <= cone_deg and d_mm > 100.0:
                if best is None or d_mm < best:
                    best = d_mm
        return best

    def _prime(self, direction):
        """Brief pulse + neutral so the ESC engages `direction` instead of
        reading the first reversed pulse as a brake."""
        thr = math.copysign(abs(BACKUP_THROTTLE), direction)
        t0 = time.monotonic()
        while time.monotonic() - t0 < BACKUP_BRAKE_PULSE_S:
            if self._stop_evt.is_set():
                return False
            self._cmd(0.0, thr)
            if self._stop_evt.wait(0.04):
                return False
        self._cmd(0.0, 0.0)
        if self._stop_evt.wait(BACKUP_RESET_NEUTRAL_S):
            return False
        return True

    def _nudge(self, direction):
        """Short clearance move (~BACKUP_DRIVE_S) in `direction` (+1 fwd,
        -1 rev), brake-primed first. Used to back off a wall/obstacle."""
        if not self._prime(direction):
            return False
        thr = math.copysign(abs(BACKUP_THROTTLE), direction)
        t0 = time.monotonic()
        while time.monotonic() - t0 < BACKUP_DRIVE_S:
            if self._stop_evt.is_set():
                self._cmd(0.0, 0.0)
                return False
            self._cmd(0.0, thr)
            if self._stop_evt.wait(0.04):
                self._cmd(0.0, 0.0)
                return False
        self._cmd(0.0, 0.0)
        if self._stop_evt.wait(0.3):
            return False
        return True

    def _fail(self, msg):
        _log(msg)
        RUNLOG.log('fail', reason=msg)
        try:
            self.driver.stop()
        except Exception:
            pass
        with self._lock:
            self.state = 'error'
            self.error = msg
