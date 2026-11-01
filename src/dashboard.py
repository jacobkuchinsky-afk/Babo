#!/usr/bin/env python3
"""DeepRacer dashboard — direct-PWM drive + live camera + telemetry + SLAM."""
import os, sys, time, signal, threading, subprocess, logging
from flask import Flask, request, jsonify, Response, send_file
import io
import cv2

# Werkzeug logs every HTTP request — at ~30 req/s during SLAM that's a lot of
# string formatting + synchronous I/O to dashboard.log on a 1.3 GHz Atom.
# Disable it entirely; errors still surface via the propagated Flask logger.
logging.getLogger('werkzeug').setLevel(logging.ERROR)
try:
    import serial as pyserial
except ImportError:
    pyserial = None

sys.path.insert(0, '/home/deepracer')
try:
    from slam_module import SlamManager, MAP_DIR as SLAM_MAP_DIR
except ImportError as _slam_err:
    SlamManager = None
    SLAM_MAP_DIR = '/home/deepracer/slam/maps'
    print(f'SLAM unavailable: {_slam_err}', file=sys.stderr, flush=True)
try:
    from auto_module import AutoNav
except ImportError as _auto_err:
    AutoNav = None
    print(f'AutoNav unavailable: {_auto_err}', file=sys.stderr, flush=True)
import json as _json

# ---------- PWM ---------------------------------------------------------------
def _find_motor_pwmchip():
    """Locate the i2c-INT3492 PWM controller — the one driving motors/steering.
    Its sysfs pwmchip number is not stable across reboots; the PCI PWM at
    pci/0000:00:1a.0 sometimes claims pwmchip0 first."""
    base = '/sys/class/pwm'
    try:
        for entry in sorted(os.listdir(base)):
            if not entry.startswith('pwmchip'):
                continue
            try:
                dev_link = os.path.realpath(f'{base}/{entry}/device')
                if 'INT3492' in dev_link:
                    return f'{base}/{entry}'
            except OSError:
                pass
    except OSError:
        pass
    # Fallback
    return '/sys/class/pwm/pwmchip0'

PWM_BASE = _find_motor_pwmchip()
print(f'Motor PWM controller: {PWM_BASE}', flush=True)
PWM_PERIOD_NS = 20_000_000

THROTTLE_PWM = 'pwm0'
THROTTLE_NEUTRAL = 1_450_000
THROTTLE_FWD_MAX = 1_380_000      # 100%% forward default (bench-verified 2026-05-28); below neutral
THROTTLE_REV_MAX = 1_520_000      # 100%% reverse default (bench-verified); above neutral
ALLOW_REVERSE = True

# When the commanded direction flips through neutral, the ESC needs a brief
# neutral hold so it interprets the first opposite-direction pulse as a brake,
# then we resume the actual command. Total transition: ~200ms.
DIR_FLIP_HOLD_MS = 200

STEER_PWM = 'pwm1'
STEER_MID = 1_430_000
STEER_MAX = 1_700_000
STEER_MIN = 1_210_000

PUB_HZ = 50
WATCHDOG_MS = 500
ARM_HOLD_S = 4.0
SERVO_GPIO = '/sys/class/gpio/gpio436/value'

SPEED_CAP_FILE = '/home/deepracer/.dashboard_speed_cap'
SPEED_CAP_DEFAULT = 0.80   # 80% of max

# Motor "power" is a multiplier on the active PWM range. 1.0 keeps the
# conservative defaults (FWD_MAX/REV_MAX above). Above 1.0 stretches the range
# toward the stock-Amazon calibration ceiling (1280000 fwd / 1580000 rev) —
# useful when the bot is loaded down and needs more punch. Below 1.0 tames the
# motor (safer for testing).
MOTOR_POWER_FILE = '/home/deepracer/.dashboard_motor_power'
MOTOR_POWER_DEFAULT = 1.0
MOTOR_POWER_MIN = 0.5
MOTOR_POWER_MAX = 1.5
# Hard ceilings from Amazon's stock calibration.json — past these the ESC may
# misbehave or refuse the pulse.
SAFE_FWD_LIMIT_NS = 1_280_000   # 150%/max forward ceiling (bench: very top it should go)
SAFE_REV_LIMIT_NS = 1_580_000   # 150%/max reverse ceiling

# Steering trim — corrects mechanical drift when angle=0 doesn't actually
# point the wheels straight. Stored as a fraction in [-STEER_TRIM_MAX,
# +STEER_TRIM_MAX]. The trim is added to the user's angle input before the
# PWM mapping; positive = the bot acts as if you're holding a small right
# turn (use this to cancel a *left* drift), negative = cancels a right drift.
STEER_TRIM_FILE = '/home/deepracer/.dashboard_steer_trim'
STEER_TRIM_DEFAULT = 0.0
STEER_TRIM_MAX = 0.50   # ±50% of full deflection — strong enough to fully counter mechanical drift

# Velocity calibration — m/s the bot reaches at driver.throttle == 1.0 and
# motor_power == 1.0. Used to publish /cmd_vel so slam_toolbox knows how fast
# we're commanding the bot. Too high → ghost parallel walls during mapping
# (odometry overshoots, scan-matching can't pull back). Too low → SLAM ignores
# real motion and the map drags behind. Tune via the slider.
VEL_FACTOR_FILE = '/home/deepracer/.dashboard_vel_factor'
VEL_FACTOR_DEFAULT = 0.6   # conservative — better to start too slow than too fast
VEL_FACTOR_MIN = 0.1
VEL_FACTOR_MAX = 3.0


def _load_speed_cap():
    try:
        with open(SPEED_CAP_FILE) as f:
            v = float(f.read().strip())
        if 0.05 <= v <= 1.0:
            return v
    except (OSError, ValueError):
        pass
    return SPEED_CAP_DEFAULT


def _save_speed_cap(v):
    try:
        with open(SPEED_CAP_FILE, 'w') as f:
            f.write(f'{v:.4f}\n')
    except OSError as e:
        print(f'speed cap save failed: {e}', file=sys.stderr)


def _load_motor_power():
    try:
        with open(MOTOR_POWER_FILE) as f:
            v = float(f.read().strip())
        if MOTOR_POWER_MIN <= v <= MOTOR_POWER_MAX:
            return v
    except (OSError, ValueError):
        pass
    return MOTOR_POWER_DEFAULT


def _save_motor_power(v):
    try:
        with open(MOTOR_POWER_FILE, 'w') as f:
            f.write(f'{v:.4f}\n')
    except OSError as e:
        print(f'motor power save failed: {e}', file=sys.stderr)


def _load_vel_factor():
    try:
        with open(VEL_FACTOR_FILE) as f:
            v = float(f.read().strip())
        if VEL_FACTOR_MIN <= v <= VEL_FACTOR_MAX:
            return v
    except (OSError, ValueError):
        pass
    return VEL_FACTOR_DEFAULT


def _save_vel_factor(v):
    try:
        with open(VEL_FACTOR_FILE, 'w') as f:
            f.write(f'{v:.4f}\n')
    except OSError as e:
        print(f'vel factor save failed: {e}', file=sys.stderr)


def _load_steer_trim():
    try:
        with open(STEER_TRIM_FILE) as f:
            v = float(f.read().strip())
        if -STEER_TRIM_MAX <= v <= STEER_TRIM_MAX:
            return v
    except (OSError, ValueError):
        pass
    return STEER_TRIM_DEFAULT


def _save_steer_trim(v):
    try:
        with open(STEER_TRIM_FILE, 'w') as f:
            f.write(f'{v:.4f}\n')
    except OSError as e:
        print(f'steer trim save failed: {e}', file=sys.stderr)


def _pwm_ranges_for(power):
    """Given a motor_power factor (0.5..1.5), return (fwd_range_ns, rev_range_ns).

    fwd_range is "ns below neutral at full throttle", rev_range is "above".
    At power=1.0, returns the conservative defaults (THROTTLE_FWD_MAX/REV_MAX).
    At power=1.5, returns the stock-cal max (SAFE_FWD_LIMIT_NS/SAFE_REV_LIMIT_NS).
    Below 1.0, scales the conservative range linearly."""
    p = max(MOTOR_POWER_MIN, min(MOTOR_POWER_MAX, power))
    cur_fwd = THROTTLE_NEUTRAL - THROTTLE_FWD_MAX        # 70_000
    cur_rev = THROTTLE_REV_MAX - THROTTLE_NEUTRAL        # 70_000
    if p <= 1.0:
        return int(cur_fwd * p), int(cur_rev * p)
    max_fwd = THROTTLE_NEUTRAL - SAFE_FWD_LIMIT_NS       # 170_000
    max_rev = SAFE_REV_LIMIT_NS - THROTTLE_NEUTRAL       # 130_000
    t = (p - 1.0) / (MOTOR_POWER_MAX - 1.0)              # 0..1 from 100%→150%
    fwd = cur_fwd + t * (max_fwd - cur_fwd)
    rev = cur_rev + t * (max_rev - cur_rev)
    return int(fwd), int(rev)


def _pwm_write(channel, attr, value):
    with open(f'{PWM_BASE}/{channel}/{attr}', 'w') as f:
        f.write(str(value))


def _pwm_export(channel_index):
    path = f'{PWM_BASE}/pwm{channel_index}'
    if os.path.isdir(path):
        return
    try:
        with open(f'{PWM_BASE}/export', 'w') as f:
            f.write(str(channel_index))
    except OSError as e:
        print(f'pwm export {channel_index} failed: {e}', file=sys.stderr)


def _pwm_init(channel, neutral):
    # Make sure the channel is exported (sysfs files exist) before writing.
    for _ in range(30):
        if os.path.isdir(f'{PWM_BASE}/{channel}'):
            break
        time.sleep(0.1)
    # Don't toggle enable — pausing the pulse train can put the ESC back into
    # arming mode. Just set period and duty_cycle, then enable if needed.
    try:
        with open(f'{PWM_BASE}/{channel}/enable') as f:
            already_on = f.read().strip() == '1'
    except OSError:
        already_on = False
    try: _pwm_write(channel, 'period', PWM_PERIOD_NS)
    except OSError: pass
    _pwm_write(channel, 'duty_cycle', neutral)
    if not already_on:
        _pwm_write(channel, 'enable', 1)


STOP_LATCH_MS = 1000   # /stop forces neutral for this long, ignoring incoming /drive


class Driver:
    def __init__(self):
        # GPIO 436 is active_low: value=0 enables the servo/ESC power.
        try:
            with open(SERVO_GPIO, 'w') as f: f.write('0')
        except OSError as e:
            print(f'servo gpio write failed: {e}', file=sys.stderr)
        _pwm_export(0); _pwm_export(1)
        _pwm_init(STEER_PWM, STEER_MID)
        _pwm_init(THROTTLE_PWM, THROTTLE_NEUTRAL)
        print(f'arming ESC: holding throttle neutral {ARM_HOLD_S}s…', flush=True)
        time.sleep(ARM_HOLD_S)
        self.lock = threading.Lock()
        self.angle = 0.0
        self.throttle = 0.0
        self.speed_cap = _load_speed_cap()
        self.motor_power = _load_motor_power()
        self.steer_trim = _load_steer_trim()
        self.vel_factor = _load_vel_factor()
        self.last_cmd_ns = time.monotonic_ns()
        # Safety: STOP_LATCH forces neutral for STOP_LATCH_MS after /stop is
        # called, defeating the bounce where a still-held key/joystick on the
        # client immediately re-applies a non-zero throttle.
        self._stop_latch_until_ns = 0
        # KILL: disable PWM pulses + ESC power line. Cleared by rearm().
        self._killed = False
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def set_cmd(self, angle, throttle):
        angle = max(-1.0, min(1.0, float(angle)))
        throttle = float(throttle)
        if not ALLOW_REVERSE:
            throttle = max(0.0, throttle)
        throttle = max(-1.0, min(1.0, throttle))
        now_ns = time.monotonic_ns()
        with self.lock:
            if self._killed:
                return  # ignore — must rearm first
            if now_ns < self._stop_latch_until_ns:
                # Latched: incoming command is forced to neutral so a still-held
                # input on the client cannot re-trigger movement immediately
                # after /stop. Update last_cmd_ns so the watchdog stays happy.
                self.angle = 0.0
                self.throttle = 0.0
                self.last_cmd_ns = now_ns
                return
            self.angle = angle
            # Scale by speed cap — cap=1.0 -> full throttle; cap=0.5 -> half.
            self.throttle = throttle * self.speed_cap
            self.last_cmd_ns = now_ns

    def set_speed_cap(self, cap):
        cap = max(0.05, min(1.0, float(cap)))
        with self.lock:
            self.speed_cap = cap
        _save_speed_cap(cap)
        return cap

    def get_speed_cap(self):
        with self.lock:
            return self.speed_cap

    def set_motor_power(self, p):
        p = max(MOTOR_POWER_MIN, min(MOTOR_POWER_MAX, float(p)))
        with self.lock:
            self.motor_power = p
        _save_motor_power(p)
        return p

    def get_motor_power(self):
        with self.lock:
            return self.motor_power

    def set_steer_trim(self, t):
        t = max(-STEER_TRIM_MAX, min(STEER_TRIM_MAX, float(t)))
        with self.lock:
            self.steer_trim = t
        _save_steer_trim(t)
        return t

    def get_steer_trim(self):
        with self.lock:
            return self.steer_trim

    def set_vel_factor(self, v):
        v = max(VEL_FACTOR_MIN, min(VEL_FACTOR_MAX, float(v)))
        with self.lock:
            self.vel_factor = v
        _save_vel_factor(v)
        return v

    def get_vel_factor(self):
        with self.lock:
            return self.vel_factor

    def stop(self):
        now_ns = time.monotonic_ns()
        with self.lock:
            self.angle = 0.0; self.throttle = 0.0
            self.last_cmd_ns = now_ns
            self._stop_latch_until_ns = now_ns + STOP_LATCH_MS * 1_000_000

    def kill(self):
        """Hardware e-stop: disable PWM output + cut ESC enable line.

        Must call rearm() to restore drive capability."""
        now_ns = time.monotonic_ns()
        with self.lock:
            self._killed = True
            self.angle = 0.0; self.throttle = 0.0
            self.last_cmd_ns = now_ns
            # Long latch so even if rearm misfires, set_cmd stays neutral.
            self._stop_latch_until_ns = now_ns + 60_000_000_000  # 60s
        # Write neutral first, then disable.
        try: _pwm_write(THROTTLE_PWM, 'duty_cycle', THROTTLE_NEUTRAL)
        except OSError: pass
        try: _pwm_write(STEER_PWM, 'duty_cycle', STEER_MID)
        except OSError: pass
        try: _pwm_write(THROTTLE_PWM, 'enable', 0)
        except OSError: pass
        try:
            with open(SERVO_GPIO, 'w') as f: f.write('1')  # active_low: disable
        except OSError: pass
        print('!! KILL: PWM disabled, ESC power off', flush=True)

    def rearm(self):
        """Reverse kill(): re-enable PWM + ESC, hold neutral to re-arm.

        Blocking — takes ~ARM_HOLD_S seconds. Run from a background thread."""
        print(f'rearming ESC: holding neutral {ARM_HOLD_S}s…', flush=True)
        try:
            with open(SERVO_GPIO, 'w') as f: f.write('0')  # active_low: enable
        except OSError: pass
        try: _pwm_write(THROTTLE_PWM, 'duty_cycle', THROTTLE_NEUTRAL)
        except OSError: pass
        try: _pwm_write(STEER_PWM, 'duty_cycle', STEER_MID)
        except OSError: pass
        try: _pwm_write(THROTTLE_PWM, 'enable', 1)
        except OSError: pass
        time.sleep(ARM_HOLD_S)
        with self.lock:
            self._killed = False
            self._stop_latch_until_ns = 0
            self.angle = 0.0; self.throttle = 0.0
            self.last_cmd_ns = time.monotonic_ns()
        print('ESC rearmed', flush=True)

    def is_killed(self):
        with self.lock:
            return self._killed

    def is_stop_latched(self):
        with self.lock:
            return time.monotonic_ns() < self._stop_latch_until_ns

    def snapshot(self):
        with self.lock:
            return self.angle, self.throttle, (time.monotonic_ns() - self.last_cmd_ns) / 1e6

    def _angle_to_duty(self, a):
        # Trim is added to the command before mapping. Clamped so the user's
        # full ±1 throw is still reachable on the un-trimmed side.
        a = a + self.steer_trim
        if a > 1.0: a = 1.0
        elif a < -1.0: a = -1.0
        a = -a  # steering polarity is inverted on this car
        if a >= 0: return int(STEER_MID + a * (STEER_MAX - STEER_MID))
        return int(STEER_MID + a * (STEER_MID - STEER_MIN))

    def _throttle_to_duty(self, t):
        # Forward = below neutral, reverse = above neutral (motor polarity -1).
        # The active range is scaled by self.motor_power (see _pwm_ranges_for).
        fwd_range, rev_range = _pwm_ranges_for(self.motor_power)
        if t > 0:
            return int(THROTTLE_NEUTRAL - t * fwd_range)
        if t < 0:
            return int(THROTTLE_NEUTRAL + (-t) * rev_range)
        return THROTTLE_NEUTRAL

    def _run(self):
        period = 1.0 / PUB_HZ
        last_dir = 0   # -1 reverse, 0 neutral, +1 forward
        hold_until_ns = 0
        while not self._stop.is_set():
            with self.lock:
                age_ms = (time.monotonic_ns() - self.last_cmd_ns) / 1e6
                a, t = self.angle, self.throttle
                killed = self._killed
                stop_latched = time.monotonic_ns() < self._stop_latch_until_ns
            if killed:
                # PWM is disabled / ESC unpowered; skip all writes.
                time.sleep(period)
                continue
            if stop_latched or age_ms > WATCHDOG_MS:
                a, t = 0.0, 0.0
            cmd_dir = 1 if t > 0 else (-1 if t < 0 else 0)
            now_ns = time.monotonic_ns()
            in_hold = now_ns < hold_until_ns

            # Brake-priming: when direction flips, hold neutral briefly so
            # the ESC reads the first reversed pulse as a brake before the
            # real command. Only triggers once per flip (last_dir resets to 0).
            if not in_hold and cmd_dir != 0 and last_dir != 0 and cmd_dir != last_dir:
                hold_until_ns = now_ns + DIR_FLIP_HOLD_MS * 1_000_000
                last_dir = 0
                in_hold = True

            if in_hold:
                t_duty = THROTTLE_NEUTRAL
            else:
                t_duty = self._throttle_to_duty(t)
                last_dir = cmd_dir

            try: _pwm_write(STEER_PWM, 'duty_cycle', self._angle_to_duty(a))
            except OSError as e: print(f'steer err: {e}', file=sys.stderr)
            try: _pwm_write(THROTTLE_PWM, 'duty_cycle', t_duty)
            except OSError as e: print(f'throttle err: {e}', file=sys.stderr)
            time.sleep(period)

    def shutdown(self):
        self._stop.set()
        self._t.join(timeout=1.0)
        try:
            _pwm_write(THROTTLE_PWM, 'duty_cycle', THROTTLE_NEUTRAL)
            _pwm_write(STEER_PWM, 'duty_cycle', STEER_MID)
        except OSError: pass


# ---------- Camera ------------------------------------------------------------
CAMERA_PRESETS = {
    'low':    {'w': 320, 'h': 240, 'fps': 10, 'q': 40},
    'medium': {'w': 480, 'h': 360, 'fps': 15, 'q': 60},
    'high':   {'w': 640, 'h': 480, 'fps': 20, 'q': 75},
}
CAMERA_QUALITY_FILE = '/home/deepracer/.dashboard_cam_quality'
CAMERA_QUALITY_DEFAULT = 'medium'


def _load_cam_quality():
    try:
        with open(CAMERA_QUALITY_FILE) as f:
            v = f.read().strip()
        if v in CAMERA_PRESETS:
            return v
    except OSError:
        pass
    return CAMERA_QUALITY_DEFAULT


def _save_cam_quality(v):
    try:
        with open(CAMERA_QUALITY_FILE, 'w') as f:
            f.write(v + '\n')
    except OSError as e:
        print(f'cam quality save failed: {e}', file=sys.stderr)


class Camera:
    def __init__(self, index=None):
        # Probe device indices 0..4 — the OS renumbers when other USB devices
        # (LiDAR hub, etc.) come and go.
        candidates = [index] if index is not None else [0, 1, 2, 3, 4]
        self.cap = None
        self.dev_index = None
        for i in candidates:
            if i is None or not os.path.exists(f'/dev/video{i}'):
                continue
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    print(f'Camera: opened /dev/video{i}', flush=True)
                    self.cap = cap
                    self.dev_index = i
                    break
                cap.release()
        if self.cap is None:
            print('Camera: no working /dev/videoN found; running with no frames',
                  file=sys.stderr, flush=True)
            self.cap = cv2.VideoCapture()

        self.quality_lock = threading.Lock()
        self.quality_name = _load_cam_quality()
        self._apply_quality_locked(self.quality_name)

        self.frame_jpeg = b''
        self.frame_ts = 0.0
        self.fps = 0.0
        # Track active /stream.mjpg consumers — when zero, the encode loop
        # idles (saves ~10–15 % CPU on the Atom). One grab keeps the V4L2
        # buffer warm so the first frame isn't stale after a reconnect.
        self._clients_lock = threading.Lock()
        self.stream_clients = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _apply_quality_locked(self, name):
        preset = CAMERA_PRESETS[name]
        self.preset = preset
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, preset['w'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, preset['h'])
            self.cap.set(cv2.CAP_PROP_FPS, preset['fps'])

    def set_quality(self, name):
        if name not in CAMERA_PRESETS:
            return self.quality_name
        with self.quality_lock:
            self.quality_name = name
            self._apply_quality_locked(name)
        _save_cam_quality(name)
        print(f'Camera quality -> {name}', flush=True)
        return name

    def get_quality(self):
        with self.quality_lock:
            return self.quality_name

    def _run(self):
        last = time.monotonic(); count = 0; ema = 0.0
        while not self._stop.is_set():
            with self._clients_lock:
                active = self.stream_clients > 0
            if not active:
                # No one is watching — drain a frame at 2 Hz so the V4L2
                # ring buffer doesn't go stale, but skip the JPEG encode.
                try: self.cap.read()
                except Exception: pass
                self.fps = 0.0
                count = 0; last = time.monotonic()
                time.sleep(0.5)
                continue
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05); continue
            with self.quality_lock:
                q = self.preset['q']
                target_fps = self.preset['fps']
            ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, q])
            if ret:
                self.frame_jpeg = buf.tobytes()
                self.frame_ts = time.monotonic()
                count += 1
                now = time.monotonic()
                if now - last >= 1.0:
                    inst = count / (now - last)
                    ema = inst if ema == 0 else 0.5 * ema + 0.5 * inst
                    self.fps = ema
                    count = 0; last = now
            # Pace to target fps so dropping quality actually drops bandwidth
            sleep_s = max(0.0, (1.0 / target_fps) - 0.005)
            time.sleep(sleep_s)

    def mjpeg_generator(self):
        with self._clients_lock:
            self.stream_clients += 1
        try:
            last_ts = 0.0
            while not self._stop.is_set():
                if self.frame_ts > last_ts and self.frame_jpeg:
                    last_ts = self.frame_ts
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                           + self.frame_jpeg + b'\r\n')
                else:
                    time.sleep(0.02)
        finally:
            with self._clients_lock:
                self.stream_clients = max(0, self.stream_clients - 1)

    def shutdown(self):
        self._stop.set()
        time.sleep(0.1)
        self.cap.release()


# ---------- Telemetry ---------------------------------------------------------
class Telemetry:
    def __init__(self):
        self.cpu_pct = 0.0
        self.mem_pct = 0.0
        self.temp_c = 0.0
        self.uptime_s = 0
        self._prev = self._read_cpu_jiffies()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    @staticmethod
    def _read_cpu_jiffies():
        with open('/proc/stat') as f:
            for line in f:
                if line.startswith('cpu '):
                    parts = [int(x) for x in line.split()[1:]]
                    idle = parts[3] + parts[4]
                    total = sum(parts)
                    return idle, total
        return 0, 0

    @staticmethod
    def _read_mem_pct():
        info = {}
        with open('/proc/meminfo') as f:
            for line in f:
                k, v = line.split(':', 1)
                info[k] = int(v.strip().split()[0])
        total = info.get('MemTotal', 1)
        avail = info.get('MemAvailable', 0)
        return 100.0 * (1.0 - avail / total)

    @staticmethod
    def _read_temp():
        best = 0.0
        for n in range(8):
            p = f'/sys/class/thermal/thermal_zone{n}/temp'
            try:
                with open(p) as f:
                    best = max(best, int(f.read()) / 1000.0)
            except OSError:
                break
        return best

    @staticmethod
    def _read_uptime():
        try:
            with open('/proc/uptime') as f:
                return int(float(f.read().split()[0]))
        except OSError:
            return 0

    def _run(self):
        while not self._stop.is_set():
            idle, total = self._read_cpu_jiffies()
            di = idle - self._prev[0]
            dt = total - self._prev[1]
            if dt > 0:
                self.cpu_pct = 100.0 * (1.0 - di / dt)
            self._prev = (idle, total)
            self.mem_pct = self._read_mem_pct()
            self.temp_c = self._read_temp()
            self.uptime_s = self._read_uptime()
            time.sleep(0.5)

    def snapshot(self):
        return {
            'cpu_pct': round(self.cpu_pct, 1),
            'mem_pct': round(self.mem_pct, 1),
            'temp_c': round(self.temp_c, 1),
            'uptime_s': self.uptime_s,
        }


# ---------- LiDAR (RPLidar A1, raw protocol over CP210x) ----------------------
LIDAR_PORT = '/dev/ttyUSB0'
LIDAR_BAUD = 115200
LIDAR_ANGLE_OFFSET_DEG = 0.0   # rotate scan if device 0° != bot forward


class Lidar:
    def __init__(self, port=LIDAR_PORT, baud=LIDAR_BAUD):
        self.port = port
        self.baud = baud
        self._ser = None
        self.lock = threading.Lock()
        self.latest = []          # list of (angle_deg, dist_mm)
        self.latest_ts = 0.0
        self.scans_per_sec = 0.0
        self.available = False
        self._stop = threading.Event()
        # Paused state: when set, the read loop closes the port and idles,
        # letting another process (slam_toolbox via rplidar_ros) own /dev/ttyUSB0.
        self._paused = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _open(self):
        if pyserial is None:
            return False
        try:
            s = pyserial.Serial(self.port, self.baud, timeout=1.0)
            # DTR/RTS low so the motor power line stays enabled.
            s.dtr = False
            s.rts = False
            s.write(b'\xa5\x25'); time.sleep(0.1)   # STOP (in case prior run left it scanning)
            s.reset_input_buffer()
            s.write(b'\xa5\x21'); time.sleep(0.1)   # FORCE_SCAN
            s.read(7)                                # discard response descriptor
            self._ser = s
            self.available = True
            print(f'LiDAR: {self.port} streaming', flush=True)
            return True
        except Exception as e:
            print(f'LiDAR: open {self.port} failed: {e}', file=sys.stderr, flush=True)
            self.available = False
            return False

    def _close(self):
        try:
            if self._ser is not None:
                try: self._ser.write(b'\xa5\x25')
                except Exception: pass
                self._ser.close()
        except Exception:
            pass
        self._ser = None
        self.available = False

    def _run(self):
        backoff = 1.0
        while not self._stop.is_set():
            if self._paused.is_set():
                if self._ser is not None:
                    self._close()
                self._stop.wait(0.5)
                continue
            if self._ser is None:
                if not self._open():
                    self._stop.wait(min(backoff, 10.0))
                    backoff = min(backoff * 2, 10.0)
                    continue
                backoff = 1.0
            try:
                self._read_loop()
            except Exception as e:
                print(f'LiDAR: read error: {e}', file=sys.stderr, flush=True)
                self._close()
                self._stop.wait(1.0)

    def pause(self):
        """Close the serial port so another process can own /dev/ttyUSB0."""
        self._paused.set()
        self._close()
        with self.lock:
            self.latest = []
            self.latest_ts = 0.0
            self.scans_per_sec = 0.0

    def resume(self):
        """Resume reading scans from the serial port."""
        self._paused.clear()

    def _read_loop(self):
        buf = bytearray()
        points = []
        scans_in_window = 0
        window_start = time.monotonic()
        while not self._stop.is_set() and not self._paused.is_set():
            if len(buf) < 5:
                chunk = self._ser.read(5 - len(buf))
                if chunk:
                    buf.extend(chunk)
                if len(buf) < 5:
                    continue
            b0, b1, b2, b3, b4 = buf[0], buf[1], buf[2], buf[3], buf[4]
            sync  = b0 & 1
            nsync = (b0 >> 1) & 1
            check = b1 & 1
            if sync == nsync or check != 1:
                # misaligned — drop one byte and try again
                del buf[0]
                continue
            del buf[:5]
            angle = ((b1 >> 1) | (b2 << 7)) / 64.0
            dist  = (b3 | (b4 << 8)) / 4.0
            quality = b0 >> 2
            if sync:
                with self.lock:
                    self.latest = points
                    self.latest_ts = time.monotonic()
                points = []
                scans_in_window += 1
                now = time.monotonic()
                if now - window_start >= 1.0:
                    self.scans_per_sec = scans_in_window / (now - window_start)
                    scans_in_window = 0
                    window_start = now
            if quality > 0 and dist > 0:
                a = angle + LIDAR_ANGLE_OFFSET_DEG
                if a >= 360.0: a -= 360.0
                elif a < 0.0: a += 360.0
                points.append((a, dist))

    def snapshot(self):
        with self.lock:
            pts = self.latest
            ts = self.latest_ts
        age_ms = (time.monotonic() - ts) * 1000.0 if ts > 0 else None
        return pts, age_ms, self.scans_per_sec

    def shutdown(self):
        self._stop.set()
        self._close()


# ---------- deepracer-core watchdog (warn only) -------------------------------
def _core_watchdog():
    while True:
        time.sleep(10)
        try:
            r = subprocess.run(['systemctl', 'is-active', 'deepracer-core'],
                               capture_output=True, text=True, timeout=3)
            if r.stdout.strip() == 'active':
                print('WARN: deepracer-core is active; will overwrite PWM. '
                      'sudo systemctl stop deepracer-core', file=sys.stderr, flush=True)
        except Exception: pass


# ---------- HTTP --------------------------------------------------------------
app = Flask(__name__)
driver = None; camera = None; telemetry = None; lidar = None; slam = None

INDEX_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<title>DeepRacer</title>
<style>
 :root {
   --bg:#0a0a0b; --panel:#141416; --panel-2:#1a1a1d;
   --line:rgba(255,255,255,.10); --line-strong:rgba(255,255,255,.18);
   --fg:#f5f5f7; --muted:#8a8a92; --accent:#ffffff;
   --good:#67e394; --warn:#ffb454; --danger:#ff5d5d;
   --radius:14px;
 }
 *, *::before, *::after { box-sizing:border-box; }
 html,body { margin:0; padding:0; background:var(--bg); color:var(--fg);
   font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", system-ui, sans-serif;
   font-feature-settings:"ss01","cv11"; -webkit-font-smoothing:antialiased;
   min-height:100%;
 }
 body { overflow-y:auto; overflow-x:hidden; }
 header {
   position:sticky; top:0; z-index:10; background:var(--bg);
   display:flex; align-items:center; justify-content:space-between;
   padding:14px 18px; border-bottom:1px solid var(--line);
 }
 .brand { display:flex; align-items:center; gap:10px; }
 .brand .dot { width:8px; height:8px; border-radius:50%; background:var(--good); box-shadow:0 0 12px var(--good); }
 .brand h1 { font-size:14px; margin:0; font-weight:600; letter-spacing:.5px; text-transform:uppercase; }
 .brand .sub { color:var(--muted); font-size:12px; }
 .status { display:flex; align-items:center; gap:8px; font-size:12px; color:var(--muted); }
 .status .pill { padding:4px 10px; border:1px solid var(--line); border-radius:999px;
                 background:var(--panel); color:var(--fg); font-variant-numeric:tabular-nums; }

 main {
   display:grid; gap:14px; padding:14px;
   grid-template-columns: minmax(0,1.4fr) minmax(280px,1fr);
 }
 .card {
   background:var(--panel); border:1px solid var(--line);
   border-radius:var(--radius); position:relative; overflow:hidden;
 }
 .card .card-title {
   position:absolute; top:10px; left:14px;
   font-size:11px; letter-spacing:1.2px; color:var(--muted);
   text-transform:uppercase; font-weight:600;
   z-index:2;
 }
 .card.cam { grid-row:1 / span 2; grid-column:1; padding:0; display:flex;
   height:46vh; max-height:520px; }
 .card.cam img { width:100%; height:100%; object-fit:contain; background:#000; }
 .card.cam .overlay {
   position:absolute; right:12px; top:10px; display:flex; gap:6px; z-index:2;
 }
 .card.cam .tag {
   font-size:11px; font-variant-numeric:tabular-nums;
   padding:4px 8px; border-radius:8px;
   background:rgba(0,0,0,.45); border:1px solid var(--line-strong); color:var(--fg);
 }

 .card.lidar { grid-row:1; grid-column:2; padding:32px 14px 12px;
   display:flex; align-items:center; justify-content:center;
   height:22vh; max-height:240px; min-height:180px; }
 .card.lidar canvas {
   display:block; background:radial-gradient(circle at center, #0e0e12 0%, #050507 100%);
   border-radius:10px;
 }
 .card.lidar .overlay {
   position:absolute; right:12px; top:8px; display:flex; gap:6px; z-index:2;
 }
 .card.lidar .tag {
   font-size:11px; font-variant-numeric:tabular-nums;
   padding:3px 7px; border-radius:8px;
   background:rgba(0,0,0,.45); border:1px solid var(--line-strong); color:var(--fg);
 }
 .card.telem { grid-row:2; grid-column:2; padding:36px 14px 14px; }
 .stats { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
 .stat {
   border:1px solid var(--line); border-radius:10px;
   background:var(--panel-2); padding:12px 12px 10px;
 }
 .stat .k { font-size:10px; letter-spacing:1.2px; color:var(--muted); text-transform:uppercase; font-weight:600; }
 .stat .v { font-size:24px; font-weight:600; margin-top:4px; font-variant-numeric:tabular-nums; }
 .stat .u { color:var(--muted); font-size:13px; margin-left:3px; font-weight:400; }
 .bar { margin-top:8px; height:6px; border-radius:3px; background:rgba(255,255,255,.06); overflow:hidden; }
 .bar > i { display:block; height:100%; background:var(--accent); transition:width .3s ease; }
 .bar.good   > i { background:var(--good); }
 .bar.warn   > i { background:var(--warn); }
 .bar.danger > i { background:var(--danger); }

 /* Live SLAM minimap on the Drive tab — only visible when SLAM is active.
    grid-row 3 sits between telem and controls; collapses to 0 when hidden. */
 .card.minimap { grid-row:3; grid-column:1 / span 2;
   height:30vh; max-height:280px; padding:32px 0 0; position:relative; }
 .card.minimap .card-title { position:absolute; top:10px; left:14px;
   font-size:11px; letter-spacing:1.2px; color:var(--muted);
   text-transform:uppercase; font-weight:600; z-index:2; }
 .card.minimap .legend { position:absolute; right:14px; top:10px; display:flex; gap:10px; z-index:2;
   font-size:11px; color:var(--muted); }
 .card.minimap .legend .dot { width:8px; height:8px; border-radius:50%;
   display:inline-block; vertical-align:middle; margin-right:4px; }
 #minimapCanvas { display:block; background:#0a0a0b; width:100%; height:calc(30vh - 32px); max-height:248px; }

 .controls { grid-row:4; grid-column:1 / span 2;
   display:grid; grid-template-columns: 1fr 1fr auto; gap:14px; }
 .pad { background:var(--panel); border:1px solid var(--line); border-radius:var(--radius);
   position:relative; height:170px; display:flex; align-items:center; justify-content:center;
   user-select:none; touch-action:none; }
 .pad .card-title { color:var(--muted); }
 .pad .knob {
   width:62px; height:62px; border-radius:50%;
   background:#fff; box-shadow: 0 6px 20px rgba(255,255,255,.18), 0 0 0 1px rgba(0,0,0,.4) inset;
   position:absolute; transition:transform .04s linear; pointer-events:none;
 }
 .pad .cross { position:absolute; inset:0; pointer-events:none; }
 .pad .cross::before, .pad .cross::after {
   content:""; position:absolute; background:rgba(255,255,255,.08);
 }
 .pad .cross::before { left:50%; top:14%; bottom:14%; width:1px; transform:translateX(-.5px); }
 .pad .cross::after  { top:50%; left:14%; right:14%; height:1px; transform:translateY(-.5px); }
 .pad .readout {
   position:absolute; bottom:10px; left:14px; pointer-events:none;
   font-size:13px; color:var(--fg); font-variant-numeric:tabular-nums;
 }
 .pad .card-title { pointer-events:none; }
 .pad .readout span { color:var(--muted); margin-right:4px; }

 .estop-wrap { display:flex; flex-direction:column; justify-content:center; gap:10px; min-width:140px; }
 .speed-cap { background:var(--panel-2); border:1px solid var(--line); border-radius:10px; padding:10px 12px; }
 .speed-label { font-size:11px; letter-spacing:1.2px; color:var(--muted); text-transform:uppercase; font-weight:600; margin-bottom:8px; }
 .speed-label b { color:var(--fg); font-variant-numeric:tabular-nums; font-size:14px; letter-spacing:0; }
 .speed-cap input[type=range] { width:100%; accent-color:#fff; }
 button.estop, button.kill, button.rearm {
   font:inherit; padding:14px 14px; border-radius:var(--radius); border:1px solid var(--line-strong);
   color:#fff; font-weight:700; letter-spacing:1px; cursor:pointer; font-size:13px;
   text-transform:uppercase;
 }
 button.estop { background:linear-gradient(180deg,#3b1d1d,#1b0c0c); }
 button.estop:hover { border-color:var(--danger); box-shadow:0 0 0 1px var(--danger) inset; }
 button.kill  { background:linear-gradient(180deg,#5a0000,#2a0000); border-color:#a00; }
 button.kill:hover  { border-color:#f55; box-shadow:0 0 0 1px #f55 inset; }
 button.rearm { background:linear-gradient(180deg,#1d3b1d,#0c1b0c); border-color:#383; }
 button.rearm:hover { border-color:var(--good); box-shadow:0 0 0 1px var(--good) inset; }
 .hint { font-size:11px; color:var(--muted); text-align:center; line-height:1.4; }
 .hint b { color:var(--danger); }

 /* Killed banner — full-width overlay that's hard to miss */
 .killed-banner {
   position:fixed; top:0; left:0; right:0; z-index:100;
   background:linear-gradient(90deg,#5a0000,#2a0000);
   color:#fff; padding:10px 14px; font-size:13px; font-weight:700;
   letter-spacing:1px; text-transform:uppercase; text-align:center;
   border-bottom:2px solid #f55;
   display:none;
 }
 .killed-banner.on { display:block; }
 .killed-banner button {
   font:inherit; margin-left:14px; padding:6px 14px; border-radius:8px;
   border:1px solid var(--good); background:transparent; color:var(--good);
   cursor:pointer; font-size:11px; letter-spacing:1px;
 }
 .killed-banner button:hover { background:var(--good); color:#000; }

 /* Settings card — scrollable extension of the dashboard for toggles */
 .settings { grid-row:5; grid-column:1 / span 2;
   background:var(--panel); border:1px solid var(--line); border-radius:var(--radius);
   padding:32px 18px 18px; position:relative; }
 .settings .card-title { position:absolute; top:10px; left:14px;
   font-size:11px; letter-spacing:1.2px; color:var(--muted);
   text-transform:uppercase; font-weight:600; }
 .settings-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:14px; }
 .setting {
   background:var(--panel-2); border:1px solid var(--line); border-radius:10px;
   padding:12px 14px;
 }
 .setting .lbl { font-size:11px; letter-spacing:1.2px; color:var(--muted);
   text-transform:uppercase; font-weight:600; margin-bottom:10px; }
 .seg { display:flex; gap:6px; }
 .seg button {
   flex:1; font:inherit; padding:8px 6px; border-radius:8px;
   border:1px solid var(--line); background:transparent; color:var(--muted);
   cursor:pointer; font-size:12px; letter-spacing:.5px; text-transform:uppercase;
 }
 .seg button.on { background:#fff; color:#000; border-color:#fff; }
 .seg button:hover:not(.on) { border-color:var(--line-strong); color:var(--fg); }

 @media (max-width: 900px) {
   main { grid-template-columns:1fr; }
   .card.cam { grid-row:1; grid-column:1; height:32vh; max-height:none; }
   .card.lidar { grid-row:2; grid-column:1; height:26vh; max-height:none; }
   .card.telem { grid-row:3; grid-column:1; }
   .card.minimap { grid-row:4; grid-column:1; }
   .controls { grid-row:5; grid-column:1; grid-template-columns:1fr 1fr; }
   .estop-wrap { grid-column:1 / span 2; }
   .settings { grid-row:6; grid-column:1; }
 }

 /* Tabs */
 .tabs { display:flex; gap:6px; }
 .tabs button {
   font:inherit; padding:6px 12px; border-radius:8px;
   border:1px solid var(--line); background:transparent; color:var(--muted);
   cursor:pointer; font-size:11px; letter-spacing:1.2px; text-transform:uppercase;
   font-weight:600;
 }
 .tabs button.on { background:#fff; color:#000; border-color:#fff; }
 .tabs button:hover:not(.on) { border-color:var(--line-strong); color:var(--fg); }
 .tab-pane { display:none; }
 .tab-pane.on { display:block; }

 /* SLAM tab */
 .slam-config {
   background:var(--panel); border:1px solid var(--line); border-radius:var(--radius);
   padding:32px 14px 14px; position:relative;
   display:flex; gap:18px; align-items:center; flex-wrap:wrap;
 }
 .slam-config .card-title { position:absolute; top:10px; left:14px;
   font-size:11px; letter-spacing:1.2px; color:var(--muted);
   text-transform:uppercase; font-weight:600; }
 .slam-config .seg { display:inline-flex; gap:6px; }
 .slam-config select {
   background:var(--panel-2); border:1px solid var(--line); color:var(--fg);
   padding:8px 12px; border-radius:8px; font:inherit; font-size:13px;
   min-width:180px;
 }
 .slam-config select:disabled { opacity:0.5; cursor:not-allowed; }
 .slam-config .mode-hint { color:var(--muted); font-size:12px; margin-left:auto; max-width:50%; text-align:right; }

 .slam-main {
   display:grid; gap:14px; padding:14px;
   grid-template-rows: auto 1fr auto;
 }
 .slam-status {
   background:var(--panel); border:1px solid var(--line); border-radius:var(--radius);
   padding:32px 14px 12px; position:relative;
   display:flex; gap:18px; flex-wrap:wrap; align-items:center;
   font-size:12px; color:var(--muted);
 }
 .slam-status .card-title { position:absolute; top:10px; left:14px;
   font-size:11px; letter-spacing:1.2px; color:var(--muted);
   text-transform:uppercase; font-weight:600; }
 .slam-status .field { display:inline-flex; gap:6px; align-items:baseline; }
 .slam-status .field b { color:var(--fg); font-variant-numeric:tabular-nums; font-size:13px; }
 .slam-status .field .lbl { font-size:10px; letter-spacing:1.2px; text-transform:uppercase; }
 .slam-status .state-pill {
   padding:4px 10px; border-radius:999px; font-size:11px; font-weight:600;
   letter-spacing:.5px; text-transform:uppercase;
   border:1px solid var(--line-strong); background:var(--panel-2); color:var(--muted);
 }
 .slam-status .state-pill.idle    { color:var(--muted); }
 .slam-status .state-pill.starting,
 .slam-status .state-pill.stopping { color:var(--warn); border-color:var(--warn); }
 .slam-status .state-pill.active,
 .slam-status .state-pill.ready,
 .slam-status .state-pill.arrived { color:var(--good); border-color:var(--good); }
 .slam-status .state-pill.driving { color:#5fa8ff; border-color:#5fa8ff; }
 .slam-status .state-pill.planning { color:var(--warn); border-color:var(--warn); }
 .slam-status .state-pill.aborted { color:var(--warn); border-color:var(--warn); }
 .slam-status .state-pill.error   { color:var(--danger); border-color:var(--danger); }
 #autoCanvas { display:block; background:#0a0a0b; width:100%; height:calc(54vh - 32px); cursor:crosshair; }

 .slam-canvas-wrap {
   background:var(--panel); border:1px solid var(--line); border-radius:var(--radius);
   position:relative; overflow:hidden; min-height:54vh;
   padding:32px 0 0;
 }
 .slam-canvas-wrap .card-title { position:absolute; top:10px; left:14px;
   font-size:11px; letter-spacing:1.2px; color:var(--muted);
   text-transform:uppercase; font-weight:600; z-index:2; }
 #slamCanvas { display:block; background:#0a0a0b; width:100%; height:calc(54vh - 32px); }
 .slam-canvas-wrap .legend {
   position:absolute; right:14px; top:10px; display:flex; gap:10px; z-index:2;
   font-size:11px; color:var(--muted);
 }
 .slam-canvas-wrap .legend .dot { width:8px; height:8px; border-radius:50%; display:inline-block; vertical-align:middle; margin-right:4px; }

 .slam-controls {
   display:flex; flex-direction:column; gap:12px;
   background:var(--panel); border:1px solid var(--line); border-radius:var(--radius);
   padding:32px 14px 14px; position:relative;
 }
 .slam-controls .card-title { position:absolute; top:10px; left:14px;
   font-size:11px; letter-spacing:1.2px; color:var(--muted);
   text-transform:uppercase; font-weight:600; }
 .slam-button-row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
 .slam-save-row { display:flex; gap:10px; align-items:center; }
 .slam-controls button {
   font:inherit; padding:10px 16px; border-radius:10px;
   border:1px solid var(--line-strong); background:var(--panel-2); color:var(--fg);
   cursor:pointer; font-size:13px; letter-spacing:.5px;
 }
 .slam-controls button.primary { background:#fff; color:#000; border-color:#fff; }
 .slam-controls button.danger  { background:linear-gradient(180deg,#3b1d1d,#1b0c0c); color:#fff; }
 .slam-controls button:disabled { opacity:0.45; cursor:not-allowed; }
 .slam-controls input[type=text] {
   flex:1;
   background:var(--panel-2); border:1px solid var(--line); color:var(--fg);
   padding:12px 14px; border-radius:10px; font:inherit; font-size:14px;
   min-width:0;
 }
 .slam-controls input[type=text]:focus {
   outline:none; border-color:var(--good); box-shadow:0 0 0 1px var(--good) inset;
 }
 .slam-controls .hint { font-size:11px; color:var(--muted); }
</style></head>
<body>
<div class="killed-banner" id="killedBanner">
  ✕ MOTOR KILLED — ESC power line cut. Click REARM to restore drive capability.
  <button id="bannerRearm">REARM</button>
</div>
<header>
  <div class="brand">
    <div class="dot"></div>
    <h1>Botguy</h1>
    <span class="sub">DeepRacer · pwm0/1</span>
  </div>
  <div class="tabs" id="tabBar">
    <button data-tab="drive" class="on">Drive</button>
    <button data-tab="slam">SLAM</button>
    <button data-tab="auto">Auto</button>
  </div>
  <div class="status">
    <span class="pill" id="link">connecting…</span>
    <span class="pill" id="age">— ms</span>
  </div>
</header>

<div id="paneDrive" class="tab-pane on">

<main>
  <section class="card cam">
    <span class="card-title">Camera</span>
    <div class="overlay">
      <span class="tag" id="camFps">— fps</span>
    </div>
    <img id="stream" src="/stream.mjpg" alt="camera">
  </section>

  <section class="card lidar">
    <span class="card-title">LiDAR</span>
    <div class="overlay">
      <span class="tag" id="lidarHz">— Hz</span>
      <span class="tag" id="lidarN">— pts</span>
    </div>
    <canvas id="lidarCanvas"></canvas>
  </section>

  <section class="card minimap" id="driveMinimap" style="display:none">
    <span class="card-title">Live map (SLAM)</span>
    <div class="legend">
      <span><span class="dot" style="background:#67e394"></span>scan</span>
      <span><span class="dot" style="background:#ffb454"></span>bot</span>
      <span><span class="dot" style="background:#fff"></span>walls</span>
    </div>
    <canvas id="minimapCanvas"></canvas>
  </section>

  <section class="card telem">
    <span class="card-title">Telemetry</span>
    <div class="stats">
      <div class="stat">
        <div class="k">CPU</div>
        <div class="v"><span id="cpuV">—</span><span class="u">%</span></div>
        <div class="bar" id="cpuBar"><i style="width:0"></i></div>
      </div>
      <div class="stat">
        <div class="k">Memory</div>
        <div class="v"><span id="memV">—</span><span class="u">%</span></div>
        <div class="bar" id="memBar"><i style="width:0"></i></div>
      </div>
      <div class="stat">
        <div class="k">Temperature</div>
        <div class="v"><span id="tempV">—</span><span class="u">°C</span></div>
        <div class="bar" id="tempBar"><i style="width:0"></i></div>
      </div>
      <div class="stat">
        <div class="k">Uptime</div>
        <div class="v" id="upV" style="font-size:18px; padding-top:6px;">—</div>
      </div>
      <div class="stat">
        <div class="k">Odom forward</div>
        <div class="v"><span id="odomFtV">—</span><span class="u">ft</span></div>
        <div class="v" style="font-size:11px; color:var(--muted); margin-top:2px;"><span id="odomMV">—</span> m</div>
      </div>
      <div class="stat">
        <div class="k">Odom lateral</div>
        <div class="v"><span id="odomLatFtV">—</span><span class="u">ft</span></div>
        <div class="v" style="font-size:11px; color:var(--muted); margin-top:2px;"><span id="odomLatMV">—</span> m</div>
      </div>
      <div class="stat">
        <div class="k">Odom yaw</div>
        <div class="v"><span id="odomYawV">—</span><span class="u">°</span></div>
        <div class="v" style="font-size:11px; color:var(--muted); margin-top:2px;">EKF-fused (RF2O+IMU)</div>
      </div>
    </div>
  </section>

  <section class="controls">
    <div class="pad" id="steer">
      <span class="card-title">Steer</span>
      <div class="cross"></div>
      <div class="knob" id="steerKnob"></div>
      <div class="readout"><span>angle</span><b id="aOut">0.00</b></div>
    </div>
    <div class="pad" id="throttle">
      <span class="card-title">Throttle  (fwd ↕ rev)</span>
      <div class="cross"></div>
      <div class="knob" id="throttleKnob"></div>
      <div class="readout"><span>throttle</span><b id="tOut">0.00</b></div>
    </div>
    <div class="estop-wrap">
      <div class="speed-cap">
        <div class="speed-label">Speed cap <b id="speedVal">80</b>%</div>
        <input type="range" id="speedCap" min="10" max="100" value="80" step="5">
      </div>
      <div class="speed-cap">
        <div class="speed-label">Motor power <b id="motorPowerVal">100</b>%</div>
        <input type="range" id="motorPower" min="50" max="150" value="100" step="5">
      </div>
      <div class="speed-cap">
        <div class="speed-label">Steer trim <b id="trimVal">0</b></div>
        <input type="range" id="steerTrim" min="-50" max="50" value="0" step="1">
      </div>
      <div class="speed-cap" title="m/s the bot reaches at full throttle. Used for SLAM odometry. Too high = ghost parallel walls; too low = SLAM ignores motion.">
        <div class="speed-label">SLAM vel cal <b id="velCalVal">0.6</b> m/s</div>
        <input type="range" id="velCal" min="10" max="300" value="60" step="5">
      </div>
      <div class="speed-cap" title="Manual yaw correction applied to every global_localize result. Use when the bot's reported orientation is consistently off (e.g. always 90° clockwise of physical). Range -180° to +180°.">
        <div class="speed-label">SLAM yaw offset <b id="yawOffsetVal">0</b>°</div>
        <input type="range" id="yawOffset" min="-180" max="180" value="0" step="1">
      </div>
      <button class="estop" id="estop">■ STOP</button>
      <button class="kill" id="killBtn">✕ KILL</button>
      <button class="rearm" id="rearmBtn" style="display:none">↻ REARM</button>
      <div class="hint">WASD / arrows<br>space = stop<br><b>KILL</b> button cuts ESC (deliberate)</div>
    </div>
  </section>

  <section class="settings">
    <span class="card-title">Settings</span>
    <div class="settings-grid">
      <div class="setting">
        <div class="lbl">Camera quality</div>
        <div class="seg" id="camQualSeg">
          <button data-q="low">Low</button>
          <button data-q="medium">Med</button>
          <button data-q="high">High</button>
        </div>
      </div>
    </div>
  </section>

  <section class="card log-viewer">
    <span class="card-title">Log</span>
    <div style="position:absolute; right:14px; top:10px; display:flex; gap:8px; z-index:2;">
      <button id="logPauseBtn" style="background:var(--panel-2); color:var(--fg); border:1px solid var(--line-strong); border-radius:6px; padding:4px 10px; font-size:11px;">⏸ pause</button>
    </div>
    <pre id="logBody" style="margin:30px 12px 12px; padding:0; max-height:240px; overflow-y:auto; font-family:'SF Mono','Consolas',monospace; font-size:10px; line-height:1.4; color:#cccccc; white-space:pre-wrap; word-break:break-all;"></pre>
  </section>
</main>
</div><!-- /paneDrive -->

<div id="paneSlam" class="tab-pane">
  <main class="slam-main">
    <section class="slam-config">
      <span class="card-title">Mode</span>
      <div class="seg" id="modeSeg">
        <button data-mode="mapping" class="on">Mapping</button>
        <button data-mode="lifelong">Lifelong</button>
        <button data-mode="localization">Localization</button>
      </div>
      <select id="mapSelect" disabled title="Saved maps">
        <option value="">— pick a saved map —</option>
      </select>
      <div class="mode-hint" id="modeHint">Build a fresh map from scratch.</div>
    </section>

    <section class="slam-status">
      <span class="card-title">SLAM status</span>
      <span class="state-pill idle" id="slamState">idle</span>
      <div class="field"><span class="lbl">scan</span><b id="slamScanRate">— Hz</b></div>
      <div class="field"><span class="lbl">map</span><b id="slamMapDims">—</b></div>
      <div class="field"><span class="lbl">pose</span><b id="slamPose">—</b></div>
      <div class="field"><span class="lbl">coverage</span><b id="slamCoverage">—</b></div>
      <div class="field"><span class="lbl">confidence</span><b id="slamConfidence">—</b></div>
      <div class="field"><span class="lbl">odom</span><b id="slamOdom">—</b></div>
      <div class="field" id="slamErrWrap" style="display:none"><span class="lbl">error</span><b id="slamErr" style="color:var(--danger)">—</b></div>
    </section>

    <section class="slam-canvas-wrap">
      <span class="card-title">Map (top-down)</span>
      <div class="legend">
        <span><span class="dot" style="background:#67e394"></span>scan</span>
        <span><span class="dot" style="background:#ffb454"></span>bot</span>
        <span><span class="dot" style="background:#fff"></span>walls</span>
      </div>
      <canvas id="slamCanvas"></canvas>
    </section>

    <section class="slam-controls">
      <span class="card-title">Controls</span>
      <div class="slam-button-row">
        <button id="slamStart" class="primary">Start SLAM</button>
        <button id="slamStop" class="danger" disabled>Stop SLAM</button>
        <button id="slamWipe" disabled title="Wipe current map and start over">Wipe Map</button>
      </div>
      <div class="slam-save-row">
        <input type="text" id="slamName" placeholder="map name (leave blank to auto-name)" autocomplete="off" spellcheck="false">
        <button id="slamSave" disabled>Save Map</button>
      </div>
      <div class="slam-save-row">
        <select id="cleanSelect" title="Pick a saved map to declutter (or the live map)"><option value="">— live map —</option></select>
        <button id="slamClean" title="Post-process: remove stray dots + straighten walls into a new clean map (non-destructive)">Clean Up Map</button>
      </div>
      <div class="slam-save-row">
        <label class="hint" style="display:flex;align-items:center;gap:4px;">declutter
          <input type="range" id="cleanDeclutter" min="3" max="31" step="2" value="13" style="width:70px;"></label>
        <label class="hint" style="display:flex;align-items:center;gap:4px;">straighten
          <input type="range" id="cleanStraighten" min="5" max="60" step="1" value="18" style="width:70px;"></label>
      </div>
      <div class="hint">Drive ~1 ft/s. Wipe restarts fresh (~15 s). Click in the name field to type — keyboard drive controls pause while you type. Clean Up: type a saved map's name to clean it (saves &lt;name&gt;_clean), or leave blank to clean the live map.</div>
    </section>
  </main>
</div><!-- /paneSlam -->

<div id="paneAuto" class="tab-pane">
  <main class="slam-main">
    <section class="slam-config">
      <span class="card-title">Map</span>
      <select id="autoMapSelect" title="Saved maps">
        <option value="">— pick a saved map —</option>
      </select>
      <div class="mode-hint">Pick a saved map, then "Localize" so the bot finds itself. Once localized, click anywhere on the map to set a goal — the path appears immediately. Hit "Go" to drive it.</div>
      <div class="slam-button-row" style="margin-top:8px">
        <button id="autoLocalize" class="primary">Localize on map</button>
        <button id="autoStopSlam">Stop SLAM</button>
      </div>
    </section>

    <section class="slam-status">
      <span class="card-title">Auto status</span>
      <span class="state-pill idle" id="autoState">idle</span>
      <div class="field"><span class="lbl">SLAM</span><b id="autoSlamState">—</b></div>
      <div class="field"><span class="lbl">pose</span><b id="autoPose">—</b></div>
      <div class="field"><span class="lbl">confidence</span><b id="autoConf">—</b></div>
      <div class="field"><span class="lbl">goal</span><b id="autoGoal">—</b></div>
      <div class="field"><span class="lbl">remaining</span><b id="autoRemaining">—</b></div>
      <div class="field" id="autoErrWrap" style="display:none"><span class="lbl">error</span><b id="autoErr" style="color:var(--danger)">—</b></div>
    </section>

    <section class="slam-canvas-wrap">
      <span class="card-title">Map · click to set goal</span>
      <div class="legend">
        <span><span class="dot" style="background:#67e394"></span>scan</span>
        <span><span class="dot" style="background:#ffb454"></span>bot</span>
        <span><span class="dot" style="background:#5fa8ff"></span>path</span>
        <span><span class="dot" style="background:#ff5d5d"></span>goal</span>
      </div>
      <canvas id="autoCanvas"></canvas>
    </section>

    <section class="slam-controls">
      <span class="card-title">Drive</span>
      <div class="slam-button-row">
        <button id="autoGo" class="primary" disabled>Go</button>
        <button id="autoAbort" class="danger" disabled>Stop</button>
        <button id="autoClear" disabled>Clear goal</button>
      </div>
      <div class="hint">A* path through inflated obstacles, pure-pursuit driving. The bot auto-stops on arrival, lost localization, or "no progress" timeout. Plain <b>Drive</b> tab keyboard still works (Space = soft stop, KILL cuts ESC power).</div>
    </section>
  </main>
</div><!-- /paneAuto -->

<script>
(() => {
 // Target = what the user is asking for (joystick pos / keyboard intent).
 // angle/throttle = smoothed values actually sent + drawn. Smoothing gives
 // the controls a "real car" feel: throttle ramps up over ~0.4s, decays
 // faster when released (engine braking), steering snaps to lock smoothly.
 let targetA = 0, targetT = 0;
 let angle = 0, throttle = 0;
 const throttleScale = 1.0;

 // Tuning (units: per-second toward target, on a [-1,1] axis)
 const STEER_RATE        = 4.5;   // 0.22 s lock-to-lock
 const STEER_CENTER_RATE = 6.0;   // self-centers faster than turning
 const THROTTLE_ACCEL    = 1.8;   // 0.56 s to full throttle
 const THROTTLE_LIFT     = 3.0;   // engine-braking decay when easing off
 const THROTTLE_BRAKE    = 5.0;   // active reverse / S key is sharper

 const $ = id => document.getElementById(id);
 const aOut=$('aOut'), tOut=$('tOut'), link=$('link'), age=$('age');
 const cpuV=$('cpuV'), memV=$('memV'), tempV=$('tempV'), upV=$('upV');
 const cpuBar=$('cpuBar').firstElementChild, memBar=$('memBar').firstElementChild, tempBar=$('tempBar').firstElementChild;
 const camFps=$('camFps');

 function setKnob(padId, knobId, x, y) {
   const pad=$(padId), knob=$(knobId);
   const r = pad.getBoundingClientRect();
   const cx=r.width/2, cy=r.height/2;
   const maxR = Math.min(cx, cy) - 36;
   knob.style.left = (cx + x*maxR - 31) + 'px';
   knob.style.top  = (cy + y*maxR - 31) + 'px';
 }
 function redraw() {
   setKnob('steer','steerKnob', angle, 0);
   setKnob('throttle','throttleKnob', 0, -throttle);
   aOut.textContent = angle.toFixed(2);
   tOut.textContent = throttle.toFixed(2);
 }
 window.addEventListener('resize', redraw);
 redraw();

 function bindPad(padId, axis) {
   const pad = $(padId); let active=false;
   function update(ev) {
     const r=pad.getBoundingClientRect();
     const pt=(ev.touches && ev.touches[0]) || ev;
     if (axis==='x') {
       const dx=(pt.clientX - r.left - r.width/2)/(r.width/2 - 36);
       targetA = Math.max(-1, Math.min(1, dx));
     } else {
       const dy=(pt.clientY - r.top - r.height/2)/(r.height/2 - 36);
       targetT = Math.max(-throttleScale, Math.min(throttleScale, -dy*throttleScale));
     }
   }
   function release() { active=false; if (axis==='x') targetA=0; else targetT=0; }
   pad.addEventListener('pointerdown', e => { active=true; pad.setPointerCapture(e.pointerId); update(e); });
   pad.addEventListener('pointermove', e => { if (active) update(e); });
   pad.addEventListener('pointerup', release);
   pad.addEventListener('pointercancel', release);
 }
 bindPad('steer','x'); bindPad('throttle','y');

 const keys = {};
 let stopLatchUntil = 0;   // performance.now() timestamp; while in future, all input is forced to 0
 let killed = false;        // mirrors server state; while true, driving is disabled

 function clearAllInput() {
   // Reset held key state and joystick target so they can't immediately
   // re-apply after a stop. The joystick "active" flag is per-pad, so we also
   // null out targets here — pads will repopulate on next pointerdown.
   for (const k in keys) keys[k] = false;
   targetA = 0; targetT = 0; angle = 0; throttle = 0; redraw();
 }

 // Soft stop: ramps to neutral, latches local input for ~1.1s, server latches 1s.
 function softStop() {
   clearAllInput();
   stopLatchUntil = performance.now() + 1100;
   fetch('/stop', {method:'POST'}).catch(()=>{});
 }

 // Hardware kill: server disables PWM + ESC enable line. Bot is electrically off
 // until rearmed.
 function killStop() {
   clearAllInput();
   stopLatchUntil = performance.now() + 60000;  // long local latch too
   killed = true;
   updateSafetyUI();
   fetch('/kill', {method:'POST'}).catch(()=>{});
 }

 function rearm() {
   fetch('/rearm', {method:'POST'}).catch(()=>{});
   // Server takes ~4s to re-arm. Keep local latch in place until status confirms.
   setTimeout(() => {
     stopLatchUntil = 0;
     killed = false;
     updateSafetyUI();
   }, 4500);
 }

 // Skip global keyboard handling when the user is typing in a form control —
 // otherwise WASD would also drive the bot while typing a map name, and Space
 // would soft-stop instead of inserting a space character.
 function _isTyping(target) {
   if (!target) return false;
   const tag = target.tagName;
   return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' ||
          target.isContentEditable === true;
 }
 addEventListener('keydown', e => {
   if (_isTyping(e.target) || _isTyping(document.activeElement)) return;
   // Space = soft STOP (latched 1s, no rearm needed). Spam it freely.
   // For hardware kill (cuts ESC power), use the KILL button — deliberate
   // action, requires the explicit click.
   if (e.key === ' ' || e.code === 'Space') {
     e.preventDefault();
     softStop();
     return;
   }
   if (e.repeat) return;
   keys[e.key.toLowerCase()] = true;
 });
 addEventListener('keyup', e => {
   if (_isTyping(e.target) || _isTyping(document.activeElement)) return;
   keys[e.key.toLowerCase()] = false;
 });

 function readKbTarget() {
   const usingKb = ['w','a','s','d','arrowup','arrowdown','arrowleft','arrowright']
                    .some(k => keys[k]);
   if (!usingKb) return null;
   const a = (keys['a']||keys['arrowleft']?-1:0)+(keys['d']||keys['arrowright']?1:0);
   const t = (keys['w']||keys['arrowup']?1:0)+(keys['s']||keys['arrowdown']?-1:0);
   return { a: a * 0.85, t: t * throttleScale };
 }

 // Smoothing loop — ramps angle/throttle toward target each frame.
 let lastTickMs = performance.now();
 function tick() {
   const now = performance.now();
   const dt = Math.min(0.05, (now - lastTickMs) / 1000);
   lastTickMs = now;

   // SAFETY: while latched (post-stop) or killed, force everything to 0 and
   // skip reading any held input. This is the fix for the "held key/joystick
   // re-applies throttle immediately after STOP" bug.
   if (killed || now < stopLatchUntil) {
     targetA = 0; targetT = 0; angle = 0; throttle = 0;
     redraw();
     requestAnimationFrame(tick);
     return;
   }

   // Keyboard overrides joystick target when any key is held.
   const kb = readKbTarget();
   if (kb) { targetA = kb.a; targetT = kb.t; }

   // Steering — wheel self-centers faster than it turns.
   const dA = targetA - angle;
   const steerRate = (Math.abs(targetA) < Math.abs(angle)) ? STEER_CENTER_RATE : STEER_RATE;
   const maxStep_a = steerRate * dt;
   angle += Math.sign(dA) * Math.min(Math.abs(dA), maxStep_a);

   // Throttle — distinguish "pressing harder" vs "easing off" vs "active brake".
   const dT = targetT - throttle;
   let throttleRate;
   if (Math.sign(targetT) !== Math.sign(throttle) && targetT !== 0 && throttle !== 0) {
     throttleRate = THROTTLE_BRAKE;          // commanding opposite direction
   } else if (Math.abs(targetT) < Math.abs(throttle)) {
     throttleRate = THROTTLE_LIFT;           // easing off
   } else {
     throttleRate = THROTTLE_ACCEL;          // pressing harder
   }
   const maxStep_t = throttleRate * dt;
   throttle += Math.sign(dT) * Math.min(Math.abs(dT), maxStep_t);

   redraw();
   requestAnimationFrame(tick);
 }
 requestAnimationFrame(tick);

 $('estop').addEventListener('click', softStop);
 $('killBtn').addEventListener('click', killStop);
 $('rearmBtn').addEventListener('click', rearm);
 $('bannerRearm').addEventListener('click', rearm);

 const killedBanner = $('killedBanner');
 const killBtn = $('killBtn');
 const rearmBtn = $('rearmBtn');
 function updateSafetyUI() {
   killedBanner.classList.toggle('on', killed);
   killBtn.style.display = killed ? 'none' : '';
   rearmBtn.style.display = killed ? '' : 'none';
   // Visual indication when latched (between stop and unlatch)
   const latchedNow = !killed && performance.now() < stopLatchUntil;
   $('estop').style.opacity = latchedNow ? '0.5' : '1';
 }
 setInterval(updateSafetyUI, 200);

 async function pollSafety() {
   try {
     const r = await fetch('/safety_status'); if (!r.ok) return;
     const j = await r.json();
     // Sync with server: if server says killed, force local state to match.
     if (j.killed !== killed) {
       killed = j.killed;
       if (killed) stopLatchUntil = performance.now() + 60000;
       updateSafetyUI();
     }
   } catch(e) {}
 }
 setInterval(pollSafety, 1000); pollSafety();

 // Speed cap slider
 const speedSlider = $('speedCap'); const speedVal = $('speedVal');
 let speedSendTimer = null;
 function sendSpeedCap() {
   fetch('/speed_cap', {
     method:'POST',
     headers:{'Content-Type':'application/json'},
     body: JSON.stringify({cap: parseInt(speedSlider.value,10) / 100.0})
   });
 }
 speedSlider.addEventListener('input', () => {
   speedVal.textContent = speedSlider.value;
   clearTimeout(speedSendTimer);
   speedSendTimer = setTimeout(sendSpeedCap, 120);
 });
 // Load current value from server on boot
 fetch('/speed_cap').then(r => r.json()).then(j => {
   const pct = Math.round(j.cap * 100);
   speedSlider.value = pct; speedVal.textContent = pct;
 }).catch(()=>{});

 // Motor power slider — UI shows 50–150%, server expects 0.5–1.5
 const motorPower = $('motorPower'), motorPowerVal = $('motorPowerVal');
 let motorPowerSendTimer = null;
 function sendMotorPower() {
   fetch('/motor_power', {
     method:'POST', headers:{'Content-Type':'application/json'},
     body: JSON.stringify({power: parseInt(motorPower.value, 10) / 100.0})
   });
 }
 motorPower.addEventListener('input', () => {
   motorPowerVal.textContent = motorPower.value + '%';
   clearTimeout(motorPowerSendTimer);
   motorPowerSendTimer = setTimeout(sendMotorPower, 120);
 });
 fetch('/motor_power').then(r => r.json()).then(j => {
   const pct = Math.round(j.power * 100);
   motorPower.value = pct; motorPowerVal.textContent = pct + '%';
 }).catch(()=>{});

 // Steering trim — slider is in % units (-20..+20), server expects -0.20..+0.20.
 // Sign convention matches steering: positive trim = wheels biased right
 // (cancels a *left* drift); negative = wheels biased left (cancels right drift).
 const steerTrim = $('steerTrim'), trimVal = $('trimVal');
 let steerTrimSendTimer = null;
 function sendSteerTrim() {
   fetch('/steer_trim', {
     method:'POST', headers:{'Content-Type':'application/json'},
     body: JSON.stringify({trim: parseInt(steerTrim.value, 10) / 100.0})
   });
 }
 steerTrim.addEventListener('input', () => {
   const v = parseInt(steerTrim.value, 10);
   trimVal.textContent = (v > 0 ? '+' : '') + v;
   clearTimeout(steerTrimSendTimer);
   steerTrimSendTimer = setTimeout(sendSteerTrim, 120);
 });
 // Double-click the slider to reset to 0
 steerTrim.addEventListener('dblclick', () => {
   steerTrim.value = 0; trimVal.textContent = '0'; sendSteerTrim();
 });
 fetch('/steer_trim').then(r => r.json()).then(j => {
   const v = Math.round(j.trim * 100);
   steerTrim.value = v; trimVal.textContent = (v > 0 ? '+' : '') + v;
 }).catch(()=>{});

 // SLAM velocity calibration — slider in 0.10..3.00 m/s, stored as float on server.
 const velCal = $('velCal'), velCalVal = $('velCalVal');
 let velCalSendTimer = null;
 function sendVelCal() {
   fetch('/vel_calib', {
     method:'POST', headers:{'Content-Type':'application/json'},
     body: JSON.stringify({vel: parseInt(velCal.value, 10) / 100.0})
   });
 }
 velCal.addEventListener('input', () => {
   velCalVal.textContent = (parseInt(velCal.value, 10) / 100.0).toFixed(2);
   clearTimeout(velCalSendTimer);
   velCalSendTimer = setTimeout(sendVelCal, 120);
 });
 fetch('/vel_calib').then(r => r.json()).then(j => {
   const v = Math.round(j.vel * 100);
   velCal.value = v; velCalVal.textContent = j.vel.toFixed(2);
 }).catch(()=>{});

 // SLAM yaw offset — manual correction (-180..180°) applied to every
 // global_localize result before publishing /initialpose. Use when the bot's
 // reported orientation is persistently off from physical reality.
 const yawOffset = $('yawOffset'), yawOffsetVal = $('yawOffsetVal');
 let yawOffsetSendTimer = null;
 function sendYawOffset() {
   fetch('/slam/yaw_offset', {
     method:'POST', headers:{'Content-Type':'application/json'},
     body: JSON.stringify({degrees: parseInt(yawOffset.value, 10)})
   });
 }
 yawOffset.addEventListener('input', () => {
   yawOffsetVal.textContent = yawOffset.value;
   clearTimeout(yawOffsetSendTimer);
   yawOffsetSendTimer = setTimeout(sendYawOffset, 150);
 });
 fetch('/slam/yaw_offset').then(r => r.json()).then(j => {
   const v = Math.round(j.degrees);
   yawOffset.value = v; yawOffsetVal.textContent = v;
 }).catch(()=>{});

 // ---- Log viewer ---------------------------------------------------------
 const logBody = $('logBody');
 const logPauseBtn = $('logPauseBtn');
 let logPaused = false;
 logPauseBtn.addEventListener('click', () => {
   logPaused = !logPaused;
   logPauseBtn.textContent = logPaused ? '▶ resume' : '⏸ pause';
 });
 async function refreshLog() {
   if (logPaused) return;
   try {
     const r = await fetch('/log/tail?n=80');
     if (!r.ok) return;
     const j = await r.json();
     if (!j.lines) return;
     // Stick to bottom if already at bottom (so user can scroll up to read
     // earlier output without being yanked back).
     const wasAtBottom =
       logBody.scrollHeight - logBody.scrollTop - logBody.clientHeight < 20;
     logBody.textContent = j.lines.join('\n');
     if (wasAtBottom) logBody.scrollTop = logBody.scrollHeight;
   } catch (e) {}
 }
 setInterval(refreshLog, 1500);
 refreshLog();

 // Camera quality segmented buttons
 const camQualSeg = $('camQualSeg');
 function setCamQualUI(name) {
   for (const b of camQualSeg.querySelectorAll('button')) {
     b.classList.toggle('on', b.dataset.q === name);
   }
 }
 camQualSeg.addEventListener('click', e => {
   const btn = e.target.closest('button[data-q]');
   if (!btn) return;
   const name = btn.dataset.q;
   setCamQualUI(name);
   fetch('/cam_quality', {
     method:'POST',
     headers:{'Content-Type':'application/json'},
     body: JSON.stringify({name})
   }).then(() => {
     // Reload the MJPEG stream so the new resolution takes effect immediately
     const img = document.getElementById('stream');
     if (img) img.src = '/stream.mjpg?t=' + Date.now();
   });
 });
 fetch('/cam_quality').then(r => r.json()).then(j => setCamQualUI(j.name)).catch(()=>{});

 function fmtUp(s) {
   const h=Math.floor(s/3600), m=Math.floor((s%3600)/60), sec=s%60;
   return (h>0?h+'h ':'') + (m<10?'0':'') + m + 'm ' + (sec<10?'0':'') + sec + 's';
 }
 function setBar(el, pct, lo, hi) {
   el.style.width = Math.max(0, Math.min(100, pct)) + '%';
   const parent = el.parentElement;
   parent.classList.remove('good','warn','danger');
   if (pct >= hi) parent.classList.add('danger');
   else if (pct >= lo) parent.classList.add('warn');
   else parent.classList.add('good');
 }

 async function send() {
   try {
     const r = await fetch('/drive', { method:'POST',
       headers:{'Content-Type':'application/json'},
       body: JSON.stringify({angle, throttle}) });
     if (r.ok) {
       const j = await r.json();
       link.textContent = 'live'; link.style.color='var(--good)';
       age.textContent = j.age_ms.toFixed(0) + ' ms';
     } else { link.textContent='err '+r.status; link.style.color='var(--danger)'; }
   } catch(e) { link.textContent='offline'; link.style.color='var(--danger)'; }
 }
 // 150 ms ≈ 6.7 Hz, well under the 500 ms server watchdog. Was 66 ms (15 Hz)
 // which buried the Flask handler under POSTs and bought no real responsiveness.
 setInterval(send, 150);

 async function pollTelem() {
   try {
     const r = await fetch('/telemetry'); if (!r.ok) return;
     const j = await r.json();
     cpuV.textContent = j.cpu_pct.toFixed(0);
     memV.textContent = j.mem_pct.toFixed(0);
     tempV.textContent = j.temp_c.toFixed(0);
     upV.textContent = fmtUp(j.uptime_s);
     camFps.textContent = j.cam_fps.toFixed(0) + ' fps';
     setBar(cpuBar, j.cpu_pct, 60, 85);
     setBar(memBar, j.mem_pct, 70, 90);
     setBar(tempBar, j.temp_c, 65, 80);
   } catch(e){}
 }
 setInterval(pollTelem, 800); pollTelem();

 // ---- LiDAR ---------------------------------------------------------------
 const lcanvas = $('lidarCanvas');
 const lctx = lcanvas.getContext('2d');
 const lidarHz = $('lidarHz'), lidarN = $('lidarN');
 const MAX_RANGE_MM = 4000;          // 4 m view radius
 let lidarSize = 0;

 function fitLidar() {
   const dpr = window.devicePixelRatio || 1;
   const parent = lcanvas.parentElement;
   const rect = parent.getBoundingClientRect();
   const size = Math.max(60, Math.floor(Math.min(rect.width, rect.height)));
   if (size === lidarSize) return;
   lidarSize = size;
   lcanvas.style.width = size + 'px';
   lcanvas.style.height = size + 'px';
   lcanvas.width  = Math.round(size * dpr);
   lcanvas.height = Math.round(size * dpr);
   lctx.setTransform(dpr, 0, 0, dpr, 0, 0);
 }

 function drawLidar(points) {
   fitLidar();
   const s = lidarSize;
   const cx = s / 2, cy = s / 2;
   const scale = (s / 2 - 6) / MAX_RANGE_MM;     // px per mm
   lctx.clearRect(0, 0, s, s);
   // range rings (1 m steps)
   lctx.strokeStyle = 'rgba(255,255,255,0.07)'; lctx.lineWidth = 1;
   for (let r = 1000; r <= MAX_RANGE_MM; r += 1000) {
     lctx.beginPath(); lctx.arc(cx, cy, r * scale, 0, Math.PI * 2); lctx.stroke();
   }
   // crosshair
   lctx.strokeStyle = 'rgba(255,255,255,0.05)';
   lctx.beginPath();
   lctx.moveTo(cx, 4); lctx.lineTo(cx, s - 4);
   lctx.moveTo(4, cy); lctx.lineTo(s - 4, cy);
   lctx.stroke();
   // forward indicator (up = bot forward)
   lctx.strokeStyle = 'rgba(103,227,148,0.7)'; lctx.lineWidth = 1.5;
   lctx.beginPath(); lctx.moveTo(cx, cy); lctx.lineTo(cx, cy - 22); lctx.stroke();
   // points
   lctx.fillStyle = '#67e394';
   for (let i = 0; i < points.length; i++) {
     const a = points[i][0], d = points[i][1];
     if (d <= 0 || d > MAX_RANGE_MM) continue;
     const rad = (a - 90) * Math.PI / 180;       // 0° = up
     const x = cx + Math.cos(rad) * d * scale;
     const y = cy + Math.sin(rad) * d * scale;
     lctx.fillRect(x - 1, y - 1, 2, 2);
   }
   // bot at origin
   lctx.fillStyle = '#ffffff';
   lctx.beginPath(); lctx.arc(cx, cy, 3, 0, Math.PI * 2); lctx.fill();
 }

 requestAnimationFrame(() => { fitLidar(); drawLidar([]); });
 window.addEventListener('resize', () => { lidarSize = 0; fitLidar(); });

 async function pollLidar() {
   try {
     const r = await fetch('/lidar'); if (!r.ok) return;
     const j = await r.json();
     if (!j.available) {
       lidarHz.textContent = 'offline';
       lidarN.textContent = '— pts';
       drawLidar([]);
       return;
     }
     lidarHz.textContent = (j.rate_hz || 0).toFixed(1) + ' Hz';
     lidarN.textContent = j.count + ' pts';
     drawLidar(j.points);
   } catch(e) {}
 }
 // A1 only scans ~5.5 Hz — polling faster just resends the same frame.
 setInterval(pollLidar, 200);

 // ---- Tabs -----------------------------------------------------------------
 const panes = {drive: $('paneDrive'), slam: $('paneSlam'), auto: $('paneAuto')};
 let currentTab = 'drive';
 $('tabBar').addEventListener('click', e => {
   const btn = e.target.closest('button[data-tab]');
   if (!btn) return;
   const t = btn.dataset.tab;
   if (t === currentTab) return;
   for (const b of $('tabBar').querySelectorAll('button')) {
     b.classList.toggle('on', b.dataset.tab === t);
   }
   for (const k in panes) panes[k].classList.toggle('on', k === t);
   currentTab = t;
   if (t === 'slam') { fitSlam(); pollSlamStatus(); refreshSlamMap(); }
   if (t === 'auto') { fitAuto(); pollAutoStatus(); refreshAutoMapList(); refreshSlamMap(); }
 });

 // ---- SLAM tab -------------------------------------------------------------
 const slamCanvas = $('slamCanvas');
 const slamCtx = slamCanvas.getContext('2d');
 const slamStateEl = $('slamState'), slamScanRate = $('slamScanRate');
 const slamMapDims = $('slamMapDims'), slamPose = $('slamPose');
 const slamErrWrap = $('slamErrWrap'), slamErr = $('slamErr');
 const slamStart = $('slamStart'), slamStop = $('slamStop');
 const slamWipe = $('slamWipe');
 const slamSave = $('slamSave'), slamName = $('slamName');

 // World/view state. The map image is drawn in world coords (meters),
 // we apply a pan+zoom to fit it nicely. Scan points come in polar (deg, mm)
 // in the bot's frame; we use the bot pose to place them on the map.
 let mapImg = null;
 let mapMeta = null;  // {w,h,res,ox,oy} — origin is bottom-left in meters
 let view = {scale: 50, panX: 0, panY: 0};  // pixels per meter, plus pan offset
 // The RPLidar A1 is mounted ~90° rotated, so SLAM's pose theta is the LASER
 // heading; the robot's physical FORWARD (nose) points at theta + this offset.
 // Bot arrows are drawn along the NOSE so the display matches where the bot
 // actually drives (matches NOSE_DEG in lidar_odom.py / auto_module.py). The
 // scan overlays stay at raw theta — the scan points ARE in the laser frame.
 const NOSE_OFFSET_RAD = 92 * Math.PI / 180;
 let latestScan = [];
 let latestPose = null;  // {x,y,theta} — live, from /slam/status
 let latestConfidence = 1.0;  // 0..1; bot dot colored by this
 let pathPoints = [];    // sampled poses for breadcrumb trail
 let savedLastPose = null;  // {x,y,theta} — captured when the map was saved
 let slamCanvasW = 0, slamCanvasH = 0;

 // Map confidence 0..1 to a color: green > yellow > red
 function confColor(c) {
   if (c >= 0.7) return '#67e394';       // green — confident
   if (c >= 0.4) return '#ffb454';       // orange — degraded
   return '#ff5d5d';                      // red — lost
 }
 function confLabel(c) {
   if (c >= 0.7) return 'good';
   if (c >= 0.4) return 'degraded';
   return 'lost';
 }

 function fitSlam() {
   if (!panes.slam.classList.contains('on')) return;
   const dpr = window.devicePixelRatio || 1;
   const rect = slamCanvas.getBoundingClientRect();
   const w = Math.max(100, Math.floor(rect.width));
   const h = Math.max(100, Math.floor(rect.height));
   if (w === slamCanvasW && h === slamCanvasH) return;
   slamCanvasW = w; slamCanvasH = h;
   slamCanvas.width = Math.round(w * dpr);
   slamCanvas.height = Math.round(h * dpr);
   slamCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
 }
 window.addEventListener('resize', () => { slamCanvasW = 0; fitSlam(); });

 function worldToScreen(x, y) {
   // Center the view on the live bot if we have one, else the last-known
   // saved pose, else the world origin. Falling back gives the user a
   // recognizable position when previewing a saved map.
   const ref = latestPose || savedLastPose;
   const cx = ref ? ref.x : 0;
   const cy = ref ? ref.y : 0;
   const sx = slamCanvasW / 2 + (x - cx) * view.scale + view.panX;
   // Y flipped because canvas y grows downward but world y grows upward.
   const sy = slamCanvasH / 2 - (y - cy) * view.scale + view.panY;
   return [sx, sy];
 }

 function drawSlam() {
   fitSlam();
   if (!slamCanvasW) return;
   slamCtx.clearRect(0, 0, slamCanvasW, slamCanvasH);

   // 1m grid
   slamCtx.strokeStyle = 'rgba(255,255,255,0.05)'; slamCtx.lineWidth = 1;
   const step = view.scale;  // 1m in pixels
   const ox = ((slamCanvasW / 2 + view.panX) % step + step) % step;
   const oy = ((slamCanvasH / 2 + view.panY) % step + step) % step;
   for (let x = ox; x < slamCanvasW; x += step) {
     slamCtx.beginPath(); slamCtx.moveTo(x, 0); slamCtx.lineTo(x, slamCanvasH); slamCtx.stroke();
   }
   for (let y = oy; y < slamCanvasH; y += step) {
     slamCtx.beginPath(); slamCtx.moveTo(0, y); slamCtx.lineTo(slamCanvasW, y); slamCtx.stroke();
   }

   // Map image
   if (mapImg && mapMeta) {
     const wpx = mapMeta.w * mapMeta.res * view.scale;
     const hpx = mapMeta.h * mapMeta.res * view.scale;
     // The PNG was flipped vertically server-side so map +y is up in the image.
     // Place its bottom-left at (ox, oy) in world coords.
     const [sx, sy] = worldToScreen(mapMeta.ox, mapMeta.oy + mapMeta.h * mapMeta.res);
     slamCtx.imageSmoothingEnabled = false;
     slamCtx.globalAlpha = 0.85;
     slamCtx.drawImage(mapImg, sx, sy, wpx, hpx);
     slamCtx.globalAlpha = 1.0;
   }

   // Path trail
   if (pathPoints.length > 1) {
     slamCtx.strokeStyle = 'rgba(255,180,80,0.4)'; slamCtx.lineWidth = 1.5;
     slamCtx.beginPath();
     const [px0, py0] = worldToScreen(pathPoints[0].x, pathPoints[0].y);
     slamCtx.moveTo(px0, py0);
     for (let i = 1; i < pathPoints.length; i++) {
       const [px, py] = worldToScreen(pathPoints[i].x, pathPoints[i].y);
       slamCtx.lineTo(px, py);
     }
     slamCtx.stroke();
   }

   // Scan points (in bot's frame; rotate+translate by pose)
   if (latestPose && latestScan.length) {
     const cos = Math.cos(latestPose.theta), sin = Math.sin(latestPose.theta);
     slamCtx.fillStyle = '#67e394';
     for (let i = 0; i < latestScan.length; i++) {
       const [angDeg, distMm] = latestScan[i];
       const distM = distMm / 1000.0;
       // Convert lidar polar to local cartesian (x forward, y left).
       // Lidar 0° = bot forward.
       const rad = angDeg * Math.PI / 180;
       const lx = Math.cos(rad) * distM;
       const ly = Math.sin(rad) * distM;
       // Rotate into map frame using bot heading, then translate.
       const wx = latestPose.x + cos * lx - sin * ly;
       const wy = latestPose.y + sin * lx + cos * ly;
       const [sx, sy] = worldToScreen(wx, wy);
       slamCtx.fillRect(sx - 1, sy - 1, 2, 2);
     }
   } else if (latestScan.length) {
     // Pose unknown — draw scan around screen center oriented up.
     const [cx, cy] = [slamCanvasW / 2, slamCanvasH / 2];
     slamCtx.fillStyle = '#67e394';
     for (let i = 0; i < latestScan.length; i++) {
       const [angDeg, distMm] = latestScan[i];
       const distM = distMm / 1000.0;
       const rad = (angDeg - 90) * Math.PI / 180;
       const sx = cx + Math.cos(rad) * distM * view.scale;
       const sy = cy + Math.sin(rad) * distM * view.scale;
       slamCtx.fillRect(sx - 1, sy - 1, 2, 2);
     }
   }

   // Last-known position (when SLAM is idle and we've loaded a saved map)
   // shown as a faded gray marker so the user sees "where you were when you
   // saved" without having to start SLAM again. Skipped if we have a live pose.
   if (!latestPose && savedLastPose) {
     const [bx, by] = worldToScreen(savedLastPose.x, savedLastPose.y);
     const len = 14;
     const c = Math.cos(savedLastPose.theta + NOSE_OFFSET_RAD), s = Math.sin(savedLastPose.theta + NOSE_OFFSET_RAD);
     slamCtx.strokeStyle = 'rgba(180,180,200,0.7)'; slamCtx.lineWidth = 2;
     slamCtx.beginPath();
     slamCtx.moveTo(bx, by);
     slamCtx.lineTo(bx + c * len, by - s * len);
     slamCtx.stroke();
     slamCtx.fillStyle = 'rgba(180,180,200,0.7)';
     slamCtx.beginPath(); slamCtx.arc(bx, by, 4, 0, Math.PI * 2); slamCtx.fill();
     // Small "saved" label so the user knows this isn't live.
     slamCtx.fillStyle = 'rgba(180,180,200,0.85)';
     slamCtx.font = '10px -apple-system,system-ui,sans-serif';
     slamCtx.fillText('saved', bx + 7, by - 7);
   }

   // Bot arrow — colored by SLAM confidence (live pose)
   if (latestPose) {
     const [bx, by] = worldToScreen(latestPose.x, latestPose.y);
     const len = 14;
     const c = Math.cos(latestPose.theta + NOSE_OFFSET_RAD), s = Math.sin(latestPose.theta + NOSE_OFFSET_RAD);
     const col = confColor(latestConfidence);
     slamCtx.strokeStyle = col; slamCtx.lineWidth = 2;
     slamCtx.beginPath();
     slamCtx.moveTo(bx, by);
     // World +x with +theta CCW. Screen y flipped.
     slamCtx.lineTo(bx + c * len, by - s * len);
     slamCtx.stroke();
     slamCtx.fillStyle = col;
     slamCtx.beginPath(); slamCtx.arc(bx, by, 4, 0, Math.PI * 2); slamCtx.fill();
     // Pulsing ring when lost so the user can't miss it
     if (latestConfidence < 0.4) {
       const pulse = 6 + 4 * Math.abs(Math.sin(performance.now() / 200));
       slamCtx.strokeStyle = col; slamCtx.lineWidth = 1;
       slamCtx.beginPath(); slamCtx.arc(bx, by, pulse, 0, Math.PI * 2); slamCtx.stroke();
     }
   }
 }

 // Pan & zoom on the SLAM canvas
 {
   let dragging = false, lastX = 0, lastY = 0;
   slamCanvas.addEventListener('pointerdown', e => {
     dragging = true; lastX = e.clientX; lastY = e.clientY;
     slamCanvas.setPointerCapture(e.pointerId);
   });
   slamCanvas.addEventListener('pointermove', e => {
     if (!dragging) return;
     view.panX += e.clientX - lastX;
     view.panY += e.clientY - lastY;
     lastX = e.clientX; lastY = e.clientY;
   });
   slamCanvas.addEventListener('pointerup',     () => dragging = false);
   slamCanvas.addEventListener('pointercancel', () => dragging = false);
   slamCanvas.addEventListener('wheel', e => {
     e.preventDefault();
     const f = e.deltaY < 0 ? 1.15 : 1/1.15;
     view.scale = Math.max(5, Math.min(400, view.scale * f));
   }, {passive:false});
 }

 // Re-render at ~30Hz when SLAM tab visible
 function slamRenderLoop() {
   if (panes.slam.classList.contains('on')) drawSlam();
   requestAnimationFrame(slamRenderLoop);
 }
 requestAnimationFrame(slamRenderLoop);

 // ---- Mode + saved-map selection ------------------------------------------
 const modeSeg = $('modeSeg');
 const mapSelect = $('mapSelect');
 const modeHint = $('modeHint');
 const driveMinimap = $('driveMinimap');
 const MODE_HINTS = {
   mapping:      'Build a fresh map from scratch. Save when done.',
   lifelong:     'Build + forget. Cells not reinforced get erased — good for dynamic spaces (people walking by).',
   localization: 'Use a saved map. Bot localizes against it; the map is NOT modified.',
 };
 let selectedMode = 'mapping';
 let knownState = 'idle';

 function updateModeUI() {
   for (const b of modeSeg.querySelectorAll('button')) {
     b.classList.toggle('on', b.dataset.mode === selectedMode);
   }
   const localOnly = selectedMode === 'localization';
   // Dropdown is enabled whenever Localization is selected — even while SLAM
   // is running. Changing it while running just stages the next map; you
   // still have to Stop + Start to actually load it.
   mapSelect.disabled = !localOnly;
   mapSelect.style.opacity = localOnly ? '1' : '0.4';
   modeHint.textContent = MODE_HINTS[selectedMode] || '';
   // Mode segment buttons stay clickable when running, but only the active
   // mode's button matches the running session. Switching while active just
   // changes the *next* mode; user must stop+start to apply.
   for (const b of modeSeg.querySelectorAll('button')) b.disabled = false;
 }

 modeSeg.addEventListener('click', e => {
   const btn = e.target.closest('button[data-mode]');
   if (!btn || btn.disabled) return;
   selectedMode = btn.dataset.mode;
   updateModeUI();
   if (selectedMode === 'localization') refreshMapList();
 });

 // Preview a saved map on the SLAM canvas the moment the user picks it from
 // the dropdown — no need to start SLAM just to inspect what was saved.
 // When SLAM is already running, the new selection is just staged for the
 // next Start; we don't override the live map.
 mapSelect.addEventListener('change', () => {
   const name = mapSelect.value;
   if (!name) return;
   if (knownState === 'active' || knownState === 'starting') {
     // Don't replace the live map with a preview — but the user's selection
     // is preserved so the next Start uses it.
     return;
   }
   fetch('/slam/saved_map.png?name=' + encodeURIComponent(name))
     .then(r => {
       if (!r.ok) {
         if (r.status === 404) alert('No preview PNG saved for "' + name + '" — re-save the map to generate one.');
         return null;
       }
       const meta = {
         w:   parseInt(r.headers.get('X-Map-W') || '0', 10),
         h:   parseInt(r.headers.get('X-Map-H') || '0', 10),
         res: parseFloat(r.headers.get('X-Map-RES') || '0.05'),
         ox:  parseFloat(r.headers.get('X-Map-OX') || '0'),
         oy:  parseFloat(r.headers.get('X-Map-OY') || '0'),
       };
       // Optional last-known pose from when the map was saved.
       const px = r.headers.get('X-Pose-X');
       let saved = null;
       if (px !== null) {
         saved = {
           x: parseFloat(px),
           y: parseFloat(r.headers.get('X-Pose-Y') || '0'),
           theta: parseFloat(r.headers.get('X-Pose-Theta') || '0'),
         };
       }
       return r.blob().then(b => ({blob: b, meta, saved}));
     })
     .then(res => {
       if (!res) return;
       const img = new Image();
       img.onload = () => {
         mapImg = img; mapMeta = res.meta;
         latestPose = null; pathPoints = []; latestScan = [];
         savedLastPose = res.saved;  // null if the map predates this feature
       };
       img.src = URL.createObjectURL(res.blob);
     }).catch(()=>{});
 });

 async function refreshMapList() {
   try {
     const r = await fetch('/slam/list_maps'); if (!r.ok) return;
     const j = await r.json();
     const maps = j.maps || [];
     // Preserve current selection if possible
     const cur = mapSelect.value;
     mapSelect.innerHTML = '<option value="">' +
       (maps.length ? '— pick a saved map —' : '(no saved maps yet)') + '</option>';
     for (const m of maps) {
       const opt = document.createElement('option');
       opt.value = m.name; opt.textContent = m.name;
       if (m.name === cur) opt.selected = true;
       mapSelect.appendChild(opt);
     }
     // Mirror into the Clean Up picker (always enabled, independent of mode).
     const cs = $('cleanSelect');
     if (cs) {
       const curC = cs.value;
       cs.innerHTML = '<option value="">— live map —</option>';
       for (const m of maps) {
         const o = document.createElement('option');
         o.value = m.name; o.textContent = m.name;
         if (m.name === curC) o.selected = true;
         cs.appendChild(o);
       }
     }
   } catch(e) {}
 }
 refreshMapList();

 function setSlamState(state, err, slamInfo) {
   knownState = state;
   slamStateEl.textContent = state;
   slamStateEl.className = 'state-pill ' + state;
   const active = state === 'active';
   const idle = state === 'idle';
   const mode = (slamInfo && slamInfo.mode) || selectedMode;
   slamStart.disabled = !idle;
   slamStop.disabled  = idle || state === 'stopping';
   // In localization mode, wipe + save don't make sense (no map being built).
   const editable = active && mode !== 'localization';
   slamWipe.disabled  = !editable;
   slamSave.disabled  = !editable;
   if (err) {
     slamErrWrap.style.display = '';
     slamErr.textContent = err;
   } else {
     slamErrWrap.style.display = 'none';
   }
   updateModeUI();
   // Drive-tab minimap visibility: visible whenever SLAM is active or starting,
   // hidden when idle/stopping/error (no map to show).
   driveMinimap.style.display = (state === 'active' || state === 'starting') ? '' : 'none';
 }

 async function pollSlamStatus() {
   try {
     const r = await fetch('/slam/status'); if (!r.ok) return;
     const j = await r.json();
     setSlamState(j.state || 'idle', j.error, j);
     // Sync local selectedMode with server when SLAM is running.
     if (j.mode && j.state !== 'idle' && j.mode !== selectedMode) {
       selectedMode = j.mode;
       updateModeUI();
     }
     if (j.scan) {
       slamScanRate.textContent = (j.scan.rate_hz || 0).toFixed(1) + ' Hz · ' +
                                  (j.scan.count || 0) + ' pts';
     }
     if (j.map) {
       slamMapDims.textContent = j.map.w + '×' + j.map.h + ' @ ' +
                                  (j.map.res * 100).toFixed(1) + ' cm';
       const cov = $('slamCoverage');
       if (cov) cov.textContent = (j.map.coverage_pct ?? 0).toFixed(1) + '% · ' +
                                    (j.map.explored_m2 ?? 0).toFixed(1) + ' m²';
     } else {
       slamMapDims.textContent = '—';
       const cov = $('slamCoverage'); if (cov) cov.textContent = '—';
     }
     if (j.pose) {
       latestPose = j.pose;
       slamPose.textContent = j.pose.x.toFixed(2) + ', ' + j.pose.y.toFixed(2) +
                              ' m · ' + (j.pose.theta * 180/Math.PI).toFixed(0) + '°';
       const last = pathPoints[pathPoints.length - 1];
       if (!last || Math.hypot(last.x - j.pose.x, last.y - j.pose.y) > 0.1) {
         pathPoints.push({x: j.pose.x, y: j.pose.y});
         if (pathPoints.length > 2000) pathPoints.shift();
       }
     } else {
       slamPose.textContent = '—';
     }
     if (typeof j.confidence === 'number') {
       latestConfidence = j.confidence;
       const cE = $('slamConfidence');
       if (cE) {
         cE.textContent = confLabel(latestConfidence) + ' (' + latestConfidence.toFixed(2) + ')';
         cE.style.color = confColor(latestConfidence);
       }
     }
     // Odometry display — show absolute x/y from /odom (integrated since SLAM
     // started). Drive a known distance and verify the delta matches.
     const o = j.odom || {};
     if (typeof o.x === 'number' && typeof o.y === 'number') {
       const odomFtV = $('odomFtV'); if (odomFtV) odomFtV.textContent = (o.x * 3.281).toFixed(2);
       const odomMV = $('odomMV'); if (odomMV) odomMV.textContent = o.x.toFixed(2);
       const odomLatFtV = $('odomLatFtV'); if (odomLatFtV) odomLatFtV.textContent = (o.y * 3.281).toFixed(2);
       const odomLatMV = $('odomLatMV'); if (odomLatMV) odomLatMV.textContent = o.y.toFixed(2);
       const yawV = $('odomYawV'); if (yawV && typeof o.yaw_deg === 'number') yawV.textContent = o.yaw_deg.toFixed(1);
       const slamOdom = $('slamOdom'); if (slamOdom)
         slamOdom.textContent = o.x.toFixed(2) + ', ' + o.y.toFixed(2) + ' m  · yaw ' +
                                 (o.yaw_deg !== undefined ? o.yaw_deg.toFixed(0) + '°' : '?');
     } else {
       const ids = ['odomFtV', 'odomMV', 'odomLatFtV', 'odomLatMV', 'odomYawV', 'slamOdom'];
       for (const id of ids) { const e = $(id); if (e) e.textContent = '—'; }
     }
   } catch(e) {}
 }
 // Status polled always (cheap); used by both SLAM tab and Drive minimap.
 setInterval(pollSlamStatus, 400);

 // Scan poll: any time SLAM is active. /lidar already returns ROS scan during SLAM.
 async function pollSlamScan() {
   if (knownState !== 'active' && knownState !== 'starting') return;
   try {
     const r = await fetch('/lidar'); if (!r.ok) return;
     const j = await r.json();
     latestScan = j.points || [];
   } catch(e) {}
 }
 setInterval(pollSlamScan, 250);

 async function refreshSlamMap() {
   if (knownState !== 'active') return;
   try {
     const r = await fetch('/slam/map.png?t=' + Date.now());
     if (r.status === 204 || !r.ok) return;
     const meta = {
       w:  parseInt(r.headers.get('X-Map-W')  || '0', 10),
       h:  parseInt(r.headers.get('X-Map-H')  || '0', 10),
       res:parseFloat(r.headers.get('X-Map-Res') || '0.05'),
       ox: parseFloat(r.headers.get('X-Map-OX')  || '0'),
       oy: parseFloat(r.headers.get('X-Map-OY')  || '0'),
     };
     const blob = await r.blob();
     const img = new Image();
     img.onload = () => { mapImg = img; mapMeta = meta; };
     img.src = URL.createObjectURL(blob);
   } catch(e) {}
 }
 setInterval(refreshSlamMap, 2000);

 // ---- Drive-tab minimap rendering -------------------------------------------
 const minimapCanvas = $('minimapCanvas');
 const minimapCtx = minimapCanvas.getContext('2d');
 let minimapW = 0, minimapH = 0;
 const MINIMAP_SCALE = 30;  // pixels per meter

 function fitMinimap() {
   if (driveMinimap.style.display === 'none') return;
   const dpr = window.devicePixelRatio || 1;
   const rect = minimapCanvas.getBoundingClientRect();
   const w = Math.max(100, Math.floor(rect.width));
   const h = Math.max(100, Math.floor(rect.height));
   if (w === minimapW && h === minimapH) return;
   minimapW = w; minimapH = h;
   minimapCanvas.width = Math.round(w * dpr);
   minimapCanvas.height = Math.round(h * dpr);
   minimapCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
 }
 window.addEventListener('resize', () => { minimapW = 0; fitMinimap(); });

 function drawMinimap() {
   fitMinimap();
   if (!minimapW) return;
   const ctx = minimapCtx;
   const W = minimapW, H = minimapH;
   const scale = MINIMAP_SCALE;
   ctx.clearRect(0, 0, W, H);
   const cx = latestPose ? latestPose.x : 0;
   const cy = latestPose ? latestPose.y : 0;
   const w2s = (x, y) => [W/2 + (x - cx) * scale, H/2 - (y - cy) * scale];

   // grid
   ctx.strokeStyle = 'rgba(255,255,255,0.05)'; ctx.lineWidth = 1;
   const step = scale;
   const ox = ((W/2) % step + step) % step;
   const oy = ((H/2) % step + step) % step;
   for (let x = ox; x < W; x += step) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
   for (let y = oy; y < H; y += step) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }

   // map image
   if (mapImg && mapMeta) {
     const wpx = mapMeta.w * mapMeta.res * scale;
     const hpx = mapMeta.h * mapMeta.res * scale;
     const [sx, sy] = w2s(mapMeta.ox, mapMeta.oy + mapMeta.h * mapMeta.res);
     ctx.imageSmoothingEnabled = false;
     ctx.globalAlpha = 0.85;
     ctx.drawImage(mapImg, sx, sy, wpx, hpx);
     ctx.globalAlpha = 1.0;
   }

   // scan points (rotated by pose into map frame)
   if (latestPose && latestScan.length) {
     const cs = Math.cos(latestPose.theta), sn = Math.sin(latestPose.theta);
     ctx.fillStyle = '#67e394';
     for (let i = 0; i < latestScan.length; i++) {
       const [angDeg, distMm] = latestScan[i];
       const distM = distMm / 1000;
       const rad = angDeg * Math.PI / 180;
       const lx = Math.cos(rad) * distM, ly = Math.sin(rad) * distM;
       const wx = latestPose.x + cs * lx - sn * ly;
       const wy = latestPose.y + sn * lx + cs * ly;
       const [px, py] = w2s(wx, wy);
       ctx.fillRect(px - 1, py - 1, 2, 2);
     }
   }

   // bot arrow — colored by SLAM confidence
   if (latestPose) {
     const [bx, by] = w2s(latestPose.x, latestPose.y);
     const len = 12, c = Math.cos(latestPose.theta + NOSE_OFFSET_RAD), s = Math.sin(latestPose.theta + NOSE_OFFSET_RAD);
     const col = confColor(latestConfidence);
     ctx.strokeStyle = col; ctx.lineWidth = 2;
     ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(bx + c * len, by - s * len); ctx.stroke();
     ctx.fillStyle = col;
     ctx.beginPath(); ctx.arc(bx, by, 3.5, 0, Math.PI * 2); ctx.fill();
     if (latestConfidence < 0.4) {
       const pulse = 5 + 3 * Math.abs(Math.sin(performance.now() / 200));
       ctx.strokeStyle = col; ctx.lineWidth = 1;
       ctx.beginPath(); ctx.arc(bx, by, pulse, 0, Math.PI * 2); ctx.stroke();
     }
   }
 }

 function minimapRenderLoop() {
   if (driveMinimap.style.display !== 'none') drawMinimap();
   requestAnimationFrame(minimapRenderLoop);
 }
 requestAnimationFrame(minimapRenderLoop);

 // ---- SLAM tab buttons ------------------------------------------------------
 slamStart.addEventListener('click', async () => {
   if (selectedMode === 'localization' && !mapSelect.value) {
     alert('Pick a saved map first.');
     return;
   }
   setSlamState('starting');
   const body = {mode: selectedMode, map_file: mapSelect.value || ''};
   const r = await fetch('/slam/start', {
     method:'POST', headers:{'Content-Type':'application/json'},
     body: JSON.stringify(body)
   });
   const j = await r.json();
   if (!j.ok) alert('Start failed: ' + (j.msg || j.error));
 });
 slamStop.addEventListener('click', async () => {
   setSlamState('stopping');
   await fetch('/slam/stop', {method:'POST'});
   mapImg = null; mapMeta = null; pathPoints = []; latestScan = []; latestPose = null;
 });
 slamWipe.addEventListener('click', async () => {
   if (!confirm('Wipe the current map and start over? (~15 s)')) return;
   slamWipe.disabled = true;
   mapImg = null; mapMeta = null; pathPoints = []; latestScan = []; latestPose = null;
   await fetch('/slam/reset', {method:'POST'});
 });
 slamSave.addEventListener('click', async () => {
   let name = (slamName.value || '').trim();
   if (!name) {
     // Auto-generate something stable + sortable so the user always gets a save.
     // Format: map_YYYYMMDD_HHMMSS (server strips anything weird anyway).
     const d = new Date();
     const pad = n => String(n).padStart(2, '0');
     name = 'map_' + d.getFullYear() + pad(d.getMonth()+1) + pad(d.getDate()) +
            '_' + pad(d.getHours()) + pad(d.getMinutes()) + pad(d.getSeconds());
     slamName.value = name;  // show the user what was used
   }
   slamSave.disabled = true;
   try {
     const r = await fetch('/slam/save_map', {
       method:'POST', headers:{'Content-Type':'application/json'},
       body: JSON.stringify({name})
     });
     const j = await r.json();
     if (j.ok) {
       alert('saved as ' + name);
       refreshMapList();  // pick up the new map immediately
     } else {
       alert('save failed: ' + (j.msg || j.error));
     }
   } finally { slamSave.disabled = false; }
 });

 // Clean Up Map — post-process (denoise + straighten) into a clean map.
 // Opens the cleaned image in a new tab and saves it as <name>_clean (or
 // live_clean for the live map). Sliders tune declutter (stray-dot/finger
 // removal) and straighten (wall simplification). Non-destructive.
 const slamClean = $('slamClean');
 if (slamClean) slamClean.addEventListener('click', () => {
   const dk = $('cleanDeclutter').value, st = $('cleanStraighten').value;
   const nm = ($('cleanSelect') ? $('cleanSelect').value : '').trim();
   const p = new URLSearchParams({declutter: dk, straighten: st, t: Date.now()});
   if (nm) { p.set('name', nm); p.set('save', nm + '_clean'); }
   else { p.set('save', 'live_clean'); }
   window.open('/slam/clean_map.png?' + p.toString(), '_blank');
   // the new <name>_clean map appears in the lists after a refresh
   setTimeout(refreshMapList, 800);
 });

 // Kick off initial state read
 pollSlamStatus();
 updateModeUI();

 // ---- Auto tab ------------------------------------------------------------
 // Shares mapImg/mapMeta/latestPose/latestScan/latestConfidence with the SLAM
 // tab. Has its own canvas, pan/zoom, and click-to-set-goal handler. The
 // dashboard auto-polls /auto/status whenever the Auto tab is visible.
 const autoCanvas = $('autoCanvas');
 const autoCtx = autoCanvas.getContext('2d');
 const autoMapSelect = $('autoMapSelect');
 const autoStateEl = $('autoState');
 const autoSlamState = $('autoSlamState');
 const autoPoseEl = $('autoPose');
 const autoConfEl = $('autoConf');
 const autoGoalEl = $('autoGoal');
 const autoRemainingEl = $('autoRemaining');
 const autoErrWrap = $('autoErrWrap'), autoErr = $('autoErr');
 const autoLocalize = $('autoLocalize'), autoStopSlam = $('autoStopSlam');
 const autoGo = $('autoGo'), autoAbort = $('autoAbort'), autoClear = $('autoClear');

 let autoView = {scale: 50, panX: 0, panY: 0, initialized: false};
 let autoCanvasW = 0, autoCanvasH = 0;
 // Auto tab uses its own preview pose so saved-map previews don't conflict
 // with the live pose driven by the SLAM tab.
 let autoSavedLastPose = null;
 let autoPlan = {goal: null, path: [], state: 'idle'};

 function fitAuto() {
   if (!panes.auto.classList.contains('on')) return;
   const dpr = window.devicePixelRatio || 1;
   const rect = autoCanvas.getBoundingClientRect();
   const w = Math.max(100, Math.floor(rect.width));
   const h = Math.max(100, Math.floor(rect.height));
   if (w === autoCanvasW && h === autoCanvasH) return;
   autoCanvasW = w; autoCanvasH = h;
   autoCanvas.width = Math.round(w * dpr);
   autoCanvas.height = Math.round(h * dpr);
   autoCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
 }
 window.addEventListener('resize', () => { autoCanvasW = 0; fitAuto(); });

 function autoCenterRef() {
   // Prefer live pose; fall back to preview pose; finally world origin so the
   // canvas always has something to look at.
   if (latestPose) return latestPose;
   if (autoSavedLastPose) return autoSavedLastPose;
   if (mapMeta) return {x: mapMeta.ox + mapMeta.w * mapMeta.res / 2,
                        y: mapMeta.oy + mapMeta.h * mapMeta.res / 2};
   return {x: 0, y: 0};
 }
 function autoWorldToScreen(x, y) {
   const ref = autoCenterRef();
   const sx = autoCanvasW / 2 + (x - ref.x) * autoView.scale + autoView.panX;
   const sy = autoCanvasH / 2 - (y - ref.y) * autoView.scale + autoView.panY;
   return [sx, sy];
 }
 function autoScreenToWorld(sx, sy) {
   const ref = autoCenterRef();
   const x = ref.x + (sx - autoCanvasW / 2 - autoView.panX) / autoView.scale;
   const y = ref.y - (sy - autoCanvasH / 2 - autoView.panY) / autoView.scale;
   return [x, y];
 }

 function drawAuto() {
   fitAuto();
   if (!autoCanvasW) return;
   const W = autoCanvasW, H = autoCanvasH, ctx = autoCtx;
   ctx.clearRect(0, 0, W, H);

   // 1m grid
   ctx.strokeStyle = 'rgba(255,255,255,0.05)'; ctx.lineWidth = 1;
   const step = autoView.scale;
   const ox = ((W/2 + autoView.panX) % step + step) % step;
   const oy = ((H/2 + autoView.panY) % step + step) % step;
   for (let x = ox; x < W; x += step) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
   for (let y = oy; y < H; y += step) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }

   // Map image (shared between SLAM/Auto tabs)
   if (mapImg && mapMeta) {
     const wpx = mapMeta.w * mapMeta.res * autoView.scale;
     const hpx = mapMeta.h * mapMeta.res * autoView.scale;
     const [sx, sy] = autoWorldToScreen(mapMeta.ox, mapMeta.oy + mapMeta.h * mapMeta.res);
     ctx.imageSmoothingEnabled = false;
     ctx.globalAlpha = 0.85;
     ctx.drawImage(mapImg, sx, sy, wpx, hpx);
     ctx.globalAlpha = 1.0;
   }

   // Planned path
   if (autoPlan.path && autoPlan.path.length > 1) {
     ctx.strokeStyle = '#5fa8ff'; ctx.lineWidth = 2;
     ctx.beginPath();
     const [px0, py0] = autoWorldToScreen(autoPlan.path[0].x, autoPlan.path[0].y);
     ctx.moveTo(px0, py0);
     for (let i = 1; i < autoPlan.path.length; i++) {
       const [px, py] = autoWorldToScreen(autoPlan.path[i].x, autoPlan.path[i].y);
       ctx.lineTo(px, py);
     }
     ctx.stroke();
     // Mark waypoints
     ctx.fillStyle = '#5fa8ff';
     for (const wp of autoPlan.path) {
       const [px, py] = autoWorldToScreen(wp.x, wp.y);
       ctx.fillRect(px - 1.5, py - 1.5, 3, 3);
     }
   }

   // Goal marker
   if (autoPlan.goal) {
     const [gx, gy] = autoWorldToScreen(autoPlan.goal.x, autoPlan.goal.y);
     ctx.strokeStyle = '#ff5d5d'; ctx.fillStyle = 'rgba(255,93,93,0.25)'; ctx.lineWidth = 2;
     ctx.beginPath(); ctx.arc(gx, gy, 9, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
     // crosshair
     ctx.beginPath();
     ctx.moveTo(gx - 14, gy); ctx.lineTo(gx + 14, gy);
     ctx.moveTo(gx, gy - 14); ctx.lineTo(gx, gy + 14);
     ctx.stroke();
   }

   // Scan points (rotated into map frame using current pose)
   if (latestPose && latestScan.length) {
     const cs = Math.cos(latestPose.theta), sn = Math.sin(latestPose.theta);
     ctx.fillStyle = '#67e394';
     for (let i = 0; i < latestScan.length; i++) {
       const [angDeg, distMm] = latestScan[i];
       const distM = distMm / 1000;
       const rad = angDeg * Math.PI / 180;
       const lx = Math.cos(rad) * distM, ly = Math.sin(rad) * distM;
       const wx = latestPose.x + cs * lx - sn * ly;
       const wy = latestPose.y + sn * lx + cs * ly;
       const [px, py] = autoWorldToScreen(wx, wy);
       ctx.fillRect(px - 1, py - 1, 2, 2);
     }
   }

   // Saved-map preview pose (shown when no live pose yet)
   if (!latestPose && autoSavedLastPose) {
     const [bx, by] = autoWorldToScreen(autoSavedLastPose.x, autoSavedLastPose.y);
     const len = 14;
     const c = Math.cos(autoSavedLastPose.theta + NOSE_OFFSET_RAD), s = Math.sin(autoSavedLastPose.theta + NOSE_OFFSET_RAD);
     ctx.strokeStyle = 'rgba(180,180,200,0.7)'; ctx.lineWidth = 2;
     ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(bx + c * len, by - s * len); ctx.stroke();
     ctx.fillStyle = 'rgba(180,180,200,0.7)';
     ctx.beginPath(); ctx.arc(bx, by, 4, 0, Math.PI * 2); ctx.fill();
     ctx.fillStyle = 'rgba(180,180,200,0.85)';
     ctx.font = '10px -apple-system,system-ui,sans-serif';
     ctx.fillText('saved', bx + 7, by - 7);
   }

   // Bot arrow (live)
   if (latestPose) {
     const [bx, by] = autoWorldToScreen(latestPose.x, latestPose.y);
     const len = 14, c = Math.cos(latestPose.theta + NOSE_OFFSET_RAD), s = Math.sin(latestPose.theta + NOSE_OFFSET_RAD);
     const col = confColor(latestConfidence);
     ctx.strokeStyle = col; ctx.lineWidth = 2;
     ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(bx + c * len, by - s * len); ctx.stroke();
     ctx.fillStyle = col;
     ctx.beginPath(); ctx.arc(bx, by, 4, 0, Math.PI * 2); ctx.fill();
     if (latestConfidence < 0.4) {
       const pulse = 6 + 4 * Math.abs(Math.sin(performance.now() / 200));
       ctx.strokeStyle = col; ctx.lineWidth = 1;
       ctx.beginPath(); ctx.arc(bx, by, pulse, 0, Math.PI * 2); ctx.stroke();
     }
   }
 }

 function autoRenderLoop() {
   if (panes.auto.classList.contains('on')) drawAuto();
   requestAnimationFrame(autoRenderLoop);
 }
 requestAnimationFrame(autoRenderLoop);

 // Pan + zoom + click-to-goal. We distinguish drag from click by movement
 // threshold during pointerdown..up.
 {
   let dragging = false, downX = 0, downY = 0, lastX = 0, lastY = 0, moved = 0;
   autoCanvas.addEventListener('pointerdown', e => {
     dragging = true; downX = lastX = e.clientX; downY = lastY = e.clientY; moved = 0;
     autoCanvas.setPointerCapture(e.pointerId);
   });
   autoCanvas.addEventListener('pointermove', e => {
     if (!dragging) return;
     const dx = e.clientX - lastX, dy = e.clientY - lastY;
     autoView.panX += dx; autoView.panY += dy;
     lastX = e.clientX; lastY = e.clientY;
     moved += Math.abs(dx) + Math.abs(dy);
   });
   autoCanvas.addEventListener('pointerup', e => {
     dragging = false;
     if (moved < 4) {  // treat as click
       // Undo the pan we accidentally applied on this little movement before
       // detecting it was a click.
       autoView.panX -= (e.clientX - downX); autoView.panY -= (e.clientY - downY);
       const rect = autoCanvas.getBoundingClientRect();
       const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
       const [wx, wy] = autoScreenToWorld(sx, sy);
       sendAutoGoal(wx, wy);
     }
   });
   autoCanvas.addEventListener('pointercancel', () => { dragging = false; });
   autoCanvas.addEventListener('wheel', e => {
     e.preventDefault();
     const f = e.deltaY < 0 ? 1.15 : 1/1.15;
     autoView.scale = Math.max(5, Math.min(400, autoView.scale * f));
   }, {passive:false});
 }

 async function sendAutoGoal(x, y) {
   try {
     const r = await fetch('/auto/goal', {
       method:'POST', headers:{'Content-Type':'application/json'},
       body: JSON.stringify({x, y})
     });
     const j = await r.json();
     if (!j.ok) {
       alert('Plan failed: ' + (j.error || j.msg));
     }
     // /auto/status will fire on the next tick; nothing else to do here.
     pollAutoStatus();
   } catch (e) {
     alert('Plan request failed: ' + e);
   }
 }

 async function refreshAutoMapList() {
   try {
     const r = await fetch('/slam/list_maps'); if (!r.ok) return;
     const j = await r.json();
     const maps = j.maps || [];
     const cur = autoMapSelect.value;
     autoMapSelect.innerHTML = '<option value="">' +
       (maps.length ? '— pick a saved map —' : '(no saved maps yet)') + '</option>';
     for (const m of maps) {
       const opt = document.createElement('option');
       opt.value = m.name; opt.textContent = m.name;
       if (m.name === cur) opt.selected = true;
       autoMapSelect.appendChild(opt);
     }
   } catch (e) {}
 }

 // Preview the picked map immediately (just like the SLAM tab does).
 autoMapSelect.addEventListener('change', () => {
   const name = autoMapSelect.value;
   if (!name) return;
   if (knownState === 'active' || knownState === 'starting') {
     // SLAM is running with its own live map; don't replace it. The select is
     // just choosing which map to localize against on the next Localize click.
     return;
   }
   fetch('/slam/saved_map.png?name=' + encodeURIComponent(name)).then(r => {
     if (!r.ok) return null;
     const meta = {
       w:   parseInt(r.headers.get('X-Map-W') || '0', 10),
       h:   parseInt(r.headers.get('X-Map-H') || '0', 10),
       res: parseFloat(r.headers.get('X-Map-RES') || '0.05'),
       ox:  parseFloat(r.headers.get('X-Map-OX') || '0'),
       oy:  parseFloat(r.headers.get('X-Map-OY') || '0'),
     };
     const px = r.headers.get('X-Pose-X');
     let saved = null;
     if (px !== null) {
       saved = {x: parseFloat(px),
                y: parseFloat(r.headers.get('X-Pose-Y') || '0'),
                theta: parseFloat(r.headers.get('X-Pose-Theta') || '0')};
     }
     return r.blob().then(b => ({blob: b, meta, saved}));
   }).then(res => {
     if (!res) return;
     const img = new Image();
     img.onload = () => {
       mapImg = img; mapMeta = res.meta;
       autoSavedLastPose = res.saved;
       // Fit view to map bounds the first time we load a map in the Auto tab.
       fitAutoViewToMap();
     };
     img.src = URL.createObjectURL(res.blob);
   }).catch(()=>{});
 });

 function fitAutoViewToMap() {
   if (!mapMeta || !autoCanvasW) return;
   const worldW = mapMeta.w * mapMeta.res;
   const worldH = mapMeta.h * mapMeta.res;
   const margin = 0.85;
   const sx = (autoCanvasW * margin) / worldW;
   const sy = (autoCanvasH * margin) / worldH;
   autoView.scale = Math.max(8, Math.min(400, Math.min(sx, sy)));
   autoView.panX = 0; autoView.panY = 0;
 }

 autoLocalize.addEventListener('click', async () => {
   const map_file = autoMapSelect.value;
   if (!map_file) { alert('Pick a saved map first.'); return; }
   if (knownState === 'active' || knownState === 'starting') {
     alert('SLAM is already running. Stop it first if you want to switch maps.');
     return;
   }
   autoLocalize.disabled = true;
   try {
     const r = await fetch('/slam/start', {
       method:'POST', headers:{'Content-Type':'application/json'},
       body: JSON.stringify({mode: 'localization', map_file})
     });
     const j = await r.json();
     if (!j.ok) alert('Start failed: ' + (j.msg || j.error));
   } finally {
     setTimeout(() => { autoLocalize.disabled = false; }, 1500);
   }
 });

 autoStopSlam.addEventListener('click', async () => {
   if (knownState !== 'active' && knownState !== 'starting') return;
   // Stop the autonomous drive first so it doesn't keep firing commands at
   // a dying pose stream.
   await fetch('/auto/stop', {method:'POST'}).catch(()=>{});
   await fetch('/slam/stop', {method:'POST'}).catch(()=>{});
 });

 autoGo.addEventListener('click', async () => {
   try {
     const r = await fetch('/auto/go', {method:'POST'});
     const j = await r.json();
     if (!j.ok) alert('Go failed: ' + (j.error || j.msg));
   } catch (e) { alert('Go failed: ' + e); }
 });
 autoAbort.addEventListener('click', async () => {
   await fetch('/auto/stop', {method:'POST'}).catch(()=>{});
 });
 autoClear.addEventListener('click', async () => {
   await fetch('/auto/clear', {method:'POST'}).catch(()=>{});
 });

 function setAutoState(s, err) {
   autoStateEl.textContent = s;
   autoStateEl.className = 'state-pill ' + s;
   if (err) { autoErrWrap.style.display = ''; autoErr.textContent = err; }
   else { autoErrWrap.style.display = 'none'; }
   // Buttons
   autoGo.disabled    = !(s === 'ready');
   autoAbort.disabled = !(s === 'driving');
   autoClear.disabled = (s === 'idle' || s === 'driving');
 }

 async function pollAutoStatus() {
   try {
     const r = await fetch('/auto/status'); if (!r.ok) return;
     const j = await r.json();
     if (!j.available) {
       setAutoState('unavailable', j.error || 'auto module missing');
       return;
     }
     autoPlan.goal = j.goal || null;
     autoPlan.path = j.path || [];
     autoPlan.state = j.state;
     setAutoState(j.state || 'idle', j.error);
     autoSlamState.textContent = knownState;
     if (latestPose) {
       autoPoseEl.textContent = latestPose.x.toFixed(2) + ', ' + latestPose.y.toFixed(2) +
         ' m · ' + (latestPose.theta * 180/Math.PI).toFixed(0) + '°';
     } else {
       autoPoseEl.textContent = '—';
     }
     autoConfEl.textContent = confLabel(latestConfidence) + ' (' + latestConfidence.toFixed(2) + ')';
     autoConfEl.style.color = confColor(latestConfidence);
     autoGoalEl.textContent = j.goal
       ? j.goal.x.toFixed(2) + ', ' + j.goal.y.toFixed(2) + ' m'
       : '—';
     autoRemainingEl.textContent = (j.dist_remaining_m !== null && j.dist_remaining_m !== undefined)
       ? j.dist_remaining_m.toFixed(2) + ' m'
       : '—';
   } catch (e) {}
 }
 setInterval(() => {
   if (panes.auto.classList.contains('on')) pollAutoStatus();
 }, 400);
 // Initial map list populate so the dropdown is ready even if the user goes
 // straight to the Auto tab.
 refreshAutoMapList();
})();
</script>
</body></html>
"""

@app.route('/')
def index():
    return Response(INDEX_HTML, mimetype='text/html')

@app.route('/drive', methods=['POST'])
def drive():
    data = request.get_json(silent=True) or {}
    angle = float(data.get('angle', 0.0))
    throttle = float(data.get('throttle', 0.0))
    # The dashboard JS sends /drive at ~6.7 Hz with (0,0) as a heartbeat
    # even when the joystick is idle. Any autonomous routine that's also
    # writing to driver.set_cmd at 15 Hz (jiggle, AutoNav) gets clobbered
    # by these zeros every 150 ms — the result is motor-stall-then-pulse
    # jitter and the bot effectively can't move while the browser is open.
    # So treat the heartbeat as a no-op whenever an auto routine is in
    # control; only forward to the driver when the user actually presses
    # a key (non-trivial command), which signals user override.
    is_trivial = abs(angle) < 0.05 and abs(throttle) < 0.05
    auto_active = (auto is not None and auto.state == 'driving')
    jiggling = (slam is not None and slam.is_jiggling())
    if jiggling or auto_active:
        if is_trivial:
            a, t, age = driver.snapshot()
            return jsonify(ok=True, angle=a, throttle=t,
                           age_ms=round(age, 1),
                           deferred='auto' if auto_active else 'jiggle')
        # Real user input → cancel the auto routine and take over.
        if jiggling:
            slam.stop_jiggle()
        if auto_active:
            auto.stop()
    driver.set_cmd(angle, throttle)
    a, t, age = driver.snapshot()
    return jsonify(ok=True, angle=a, throttle=t, age_ms=round(age, 1))

@app.route('/stop', methods=['POST'])
def _stop():
    if slam is not None:
        slam.stop_jiggle()
    if auto is not None and auto.state == 'driving':
        auto.stop()
    driver.stop()
    return jsonify(ok=True, latched_ms=STOP_LATCH_MS)


@app.route('/kill', methods=['POST'])
def _kill():
    driver.kill()
    return jsonify(ok=True, killed=True)


@app.route('/rearm', methods=['POST'])
def _rearm():
    if not driver.is_killed():
        return jsonify(ok=True, msg='not killed')
    threading.Thread(target=driver.rearm, daemon=True).start()
    return jsonify(ok=True, rearming_seconds=ARM_HOLD_S)


@app.route('/safety_status')
def _safety_status():
    return jsonify(killed=driver.is_killed(),
                   stop_latched=driver.is_stop_latched())


@app.route('/log/tail')
def _log_tail():
    """Return the last N lines of dashboard.log for the in-page log viewer.
    n in query string, capped at 500, default 60. Reads only the last 50 KB
    of the file so this stays fast even when the log gets large."""
    try:
        n = int(request.args.get('n', 60))
    except ValueError:
        n = 60
    n = max(1, min(500, n))
    path = '/home/deepracer/dashboard.log'
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            chunk = min(50000, size)
            f.seek(max(0, size - chunk))
            data = f.read(chunk).decode('utf-8', errors='replace')
        lines = data.split('\n')
        # Drop the (possibly partial) first line of the chunk
        if size > chunk and len(lines) > 1:
            lines = lines[1:]
        return jsonify(lines=lines[-n:])
    except OSError as e:
        return jsonify(error=str(e)), 500

@app.route('/speed_cap', methods=['GET', 'POST'])
def speed_cap():
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        cap = driver.set_speed_cap(data.get('cap', SPEED_CAP_DEFAULT))
        return jsonify(ok=True, cap=cap)
    return jsonify(cap=driver.get_speed_cap())

@app.route('/motor_power', methods=['GET', 'POST'])
def motor_power():
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        p = driver.set_motor_power(data.get('power', MOTOR_POWER_DEFAULT))
        fwd, rev = _pwm_ranges_for(p)
        return jsonify(ok=True, power=p, fwd_range_ns=fwd, rev_range_ns=rev,
                       min=MOTOR_POWER_MIN, max=MOTOR_POWER_MAX)
    p = driver.get_motor_power()
    fwd, rev = _pwm_ranges_for(p)
    return jsonify(power=p, fwd_range_ns=fwd, rev_range_ns=rev,
                   min=MOTOR_POWER_MIN, max=MOTOR_POWER_MAX)


@app.route('/vel_calib', methods=['GET', 'POST'])
def vel_calib():
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        v = driver.set_vel_factor(data.get('vel', VEL_FACTOR_DEFAULT))
        return jsonify(ok=True, vel=v, min=VEL_FACTOR_MIN, max=VEL_FACTOR_MAX)
    return jsonify(vel=driver.get_vel_factor(),
                   min=VEL_FACTOR_MIN, max=VEL_FACTOR_MAX)


@app.route('/cmd_vx')
def cmd_vx():
    """Live commanded forward velocity (m/s) for the EKF's cmd_vel_bridge.

    vx = scaled throttle (already x speed_cap) x vel_factor x motor_power.
    Polled at ~20 Hz by cmd_vel_bridge.py running inside the SLAM launch.

    Special case: during a localization-mode jiggle we lie and report 0.
    The jiggle drives the bot at high throttle to overcome the ESC deadband,
    but vel_factor is uncalibrated so cmd-based integration over-reports the
    actual motion by 3-5×. That bad prior makes slam_toolbox search far from
    the real position. By reporting vx=0 the EKF stays "stationary" — the
    bot really is moving, but slam_toolbox's scan-matching (which has a
    ±4 m loop-search range around the EKF prior) finds the bot's actual
    location from each new scan and the live pose tracks correctly via the
    map→odom TF that slam_toolbox publishes."""
    if slam is not None and slam.is_jiggling() and slam.mode == 'localization':
        return jsonify(vx=0.0)
    _, throttle, _ = driver.snapshot()
    vx = float(throttle) * driver.get_vel_factor() * driver.get_motor_power()
    return jsonify(vx=round(vx, 4))


@app.route('/steer_trim', methods=['GET', 'POST'])
def steer_trim():
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        t = driver.set_steer_trim(data.get('trim', STEER_TRIM_DEFAULT))
        return jsonify(ok=True, trim=t, max=STEER_TRIM_MAX)
    return jsonify(trim=driver.get_steer_trim(), max=STEER_TRIM_MAX)


@app.route('/cam_quality', methods=['GET', 'POST'])
def cam_quality():
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        name = camera.set_quality(data.get('name', CAMERA_QUALITY_DEFAULT))
        return jsonify(ok=True, name=name, presets=list(CAMERA_PRESETS.keys()))
    return jsonify(name=camera.get_quality(), presets=list(CAMERA_PRESETS.keys()))

@app.route('/telemetry')
def telem():
    snap = telemetry.snapshot()
    snap['cam_fps'] = round(camera.fps, 1)
    return jsonify(snap)

@app.route('/stream.mjpg')
def stream():
    return Response(camera.mjpeg_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/lidar')
def lidar_endpoint():
    # When SLAM is active, /scan comes from rplidar_ros (the serial Lidar is
    # paused). Source from there so the existing UI keeps working seamlessly.
    if slam is not None and slam.state == 'active':
        pts, age_ms, rate = slam.scan_snapshot()
        return jsonify(
            available=True,
            source='ros',
            count=len(pts),
            age_ms=round(age_ms, 0) if age_ms is not None else None,
            rate_hz=round(rate, 1),
            points=[(round(a, 1), int(d)) for a, d in pts],
        )
    if lidar is None or not lidar.available:
        return jsonify(available=False, source='serial', count=0, rate_hz=0.0,
                       age_ms=None, points=[])
    pts, age_ms, rate = lidar.snapshot()
    return jsonify(
        available=True,
        source='serial',
        count=len(pts),
        age_ms=round(age_ms, 0) if age_ms is not None else None,
        rate_hz=round(rate, 1),
        points=[(round(a, 1), int(d)) for a, d in pts],
    )


# ---------- SLAM endpoints ----------------------------------------------------
@app.route('/slam/status')
def slam_status():
    if slam is None:
        return jsonify(available=False, state='unavailable',
                       error='slam_module not importable')
    s = slam.status()
    s['available'] = True
    return jsonify(s)


@app.route('/slam/start', methods=['POST'])
def slam_start():
    if slam is None:
        return jsonify(ok=False, error='SLAM unavailable'), 503
    data = request.get_json(silent=True) or {}
    mode = (data.get('mode') or 'mapping').lower()
    map_file = (data.get('map_file') or '').strip()
    ok, msg = slam.start(mode=mode, map_file=map_file)
    return jsonify(ok=ok, msg=msg, mode=mode, map_file=map_file)


@app.route('/slam/list_maps')
def slam_list_maps():
    if slam is None:
        return jsonify(maps=[])
    return jsonify(maps=slam.list_maps())


@app.route('/slam/stop', methods=['POST'])
def slam_stop():
    if slam is None:
        return jsonify(ok=False, error='SLAM unavailable'), 503
    ok, msg = slam.stop()
    return jsonify(ok=ok, msg=msg)


@app.route('/slam/jiggle', methods=['POST'])
def slam_jiggle():
    """Active-localization: drive a small motion sequence so the matcher can
    lock onto the saved map. JSON: {"duration_s": 8, "throttle": 0.95}."""
    if slam is None:
        return jsonify(ok=False, error='SLAM unavailable'), 503
    data = request.get_json(silent=True) or {}
    duration = float(data.get('duration_s', 8.0))
    throttle = float(data.get('throttle', 0.95))
    ok, msg = slam.jiggle_localize(duration_s=duration, throttle=throttle)
    return jsonify(ok=ok, msg=msg)


@app.route('/slam/jiggle/stop', methods=['POST'])
def slam_jiggle_stop():
    if slam is None:
        return jsonify(ok=False, error='SLAM unavailable'), 503
    stopped = slam.stop_jiggle()
    return jsonify(ok=True, stopped=stopped)


@app.route('/slam/global_localize', methods=['POST'])
def slam_global_localize():
    """Brute-force scan-vs-map search to find the bot's actual pose in the
    saved map, then publish to /initialpose so slam_toolbox adopts it.
    Optional JSON: {"sample_step_m": 0.25, "yaw_step_deg": 8.0}."""
    if slam is None:
        return jsonify(ok=False, error='SLAM unavailable'), 503
    data = request.get_json(silent=True) or {}
    step = float(data.get('sample_step_m', 0.25))
    yaw = float(data.get('yaw_step_deg', 8.0))
    pose, msg = slam.global_localize(sample_step_m=step, yaw_step_deg=yaw)
    if pose is None:
        return jsonify(ok=False, msg=msg), 200
    return jsonify(ok=True, msg=msg,
                   x=pose[0], y=pose[1], theta=pose[2])


@app.route('/slam/yaw_offset', methods=['GET', 'POST'])
def slam_yaw_offset():
    """User-tunable yaw correction (degrees, [-180, 180]) applied to every
    global_localize result before publishing /initialpose. Use to manually
    correct a persistent rotational error (e.g. always-90°-off).

    GET → {"degrees": +X.X}
    POST {"degrees": -90} to set."""
    if slam is None:
        return jsonify(ok=False, error='SLAM unavailable'), 503
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        deg = slam.set_yaw_offset_deg(data.get('degrees', 0.0))
        return jsonify(ok=True, degrees=deg)
    return jsonify(degrees=slam.get_yaw_offset_deg())


@app.route('/slam/save_map', methods=['POST'])
def slam_save_map():
    if slam is None:
        return jsonify(ok=False, error='SLAM unavailable'), 503
    data = request.get_json(silent=True) or {}
    name = (data.get('name') or '').strip()
    if not name:
        return jsonify(ok=False, error='name required'), 400
    ok, msg = slam.save_map(name)
    return jsonify(ok=ok, msg=msg)


@app.route('/slam/reset', methods=['POST'])
def slam_reset():
    """Wipe the current map by stopping + restarting the SLAM stack.

    Preserves the current mode (mapping / lifelong / localization) and map_file
    so a "Wipe" in lifelong stays lifelong, etc."""
    if slam is None:
        return jsonify(ok=False, error='SLAM unavailable'), 503
    if slam.state not in ('active', 'error'):
        return jsonify(ok=False, error=f'cannot reset from {slam.state}'), 400
    mode = slam.mode
    map_file = slam.map_file
    def cycle():
        slam.stop()
        for _ in range(50):
            if slam.state == 'idle':
                break
            time.sleep(0.2)
        slam.start(mode=mode, map_file=map_file)
    threading.Thread(target=cycle, daemon=True).start()
    return jsonify(ok=True, msg=f'resetting ({mode}, ~15s)')


@app.route('/slam/saved_map.png')
def slam_saved_map_png():
    """Serve a previously-saved map's PNG snapshot (no SLAM needed)."""
    name = (request.args.get('name') or '').strip()
    safe = ''.join(c for c in name if c.isalnum() or c in '-_')
    if not safe:
        return Response(b'', status=400)
    png_path = os.path.join(SLAM_MAP_DIR, safe + '.png')
    meta_path = os.path.join(SLAM_MAP_DIR, safe + '.meta.json')
    if not os.path.isfile(png_path):
        return Response(b'', status=404)
    try:
        with open(png_path, 'rb') as f:
            png = f.read()
    except OSError as e:
        return Response(str(e).encode(), status=500)
    resp = Response(png, mimetype='image/png')
    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                meta = _json.load(f)
            for k in ('w', 'h', 'res', 'ox', 'oy'):
                if k in meta:
                    resp.headers[f'X-Map-{k.upper()}'] = str(meta[k])
            # Optional last-known pose captured at save time, so the preview
            # can show "where you were when you stopped" even with SLAM idle.
            lp = meta.get('last_pose')
            if lp:
                resp.headers['X-Pose-X'] = str(lp.get('x', 0))
                resp.headers['X-Pose-Y'] = str(lp.get('y', 0))
                resp.headers['X-Pose-Theta'] = str(lp.get('theta', 0))
        except (OSError, ValueError):
            pass
    resp.headers['Cache-Control'] = 'no-store'
    return resp


def _clean_map_img(img, cv2, np, open_k, eps_frac):
    """Post-process an occupancy image (0=occ,255=free,127=unknown) into a
    clean map: take the largest free region, open it to shave off stray spray
    'fingers'/specks (open_k), trace its outline + internal obstacle holes, and
    simplify each to straight segments (eps_frac). Returns the cleaned image
    (free=white, walls=black, unknown=gray). See clean_map.py for the rationale."""
    free = (img > 200).astype(np.uint8)
    free = cv2.morphologyEx(free, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(free, 8)
    out = np.full_like(img, 127)
    if n <= 1:
        return out
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    room = (lab == big).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    room = cv2.morphologyEx(room, cv2.MORPH_OPEN, k)
    room = cv2.morphologyEx(room, cv2.MORPH_CLOSE, k)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(room, 8)
    if n2 > 1:
        room = (lab2 == 1 + int(np.argmax(st2[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    out[room == 1] = 255
    cnts, _ = cv2.findContours(room, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) < 60:          # drop tiny spray holes / specks
            continue
        approx = cv2.approxPolyDP(c, eps_frac * cv2.arcLength(c, True), True)
        cv2.polylines(out, [approx], True, 0, 2)
    return out


@app.route('/slam/clean_map.png')
def slam_clean_map_png():
    """Post-process a map (denoise + straighten walls) and serve it as PNG.

    Source: ?name=SAVED to clean a saved map, or the live map if omitted.
    Tunables: ?declutter=ODD_PX (free-region opening kernel, default 13; bigger
              removes more spray), ?straighten=N (polygon-simplify epsilon*1000,
              default 18 => 0.018; bigger = straighter walls).
    ?save=NAME also writes the result to maps/NAME.png (the raw map is untouched).
    """
    import numpy as np
    try:
        import cv2
    except ImportError:
        return Response(b'opencv missing', status=500)
    name = (request.args.get('name') or '').strip()
    if name:
        safe = ''.join(c for c in name if c.isalnum() or c in '-_')
        p = os.path.join(SLAM_MAP_DIR, safe + '.png')
        if not os.path.isfile(p):
            return Response(b'no such saved map', status=404)
        with open(p, 'rb') as f:
            src = f.read()
    else:
        if slam is None:
            return Response(b'', status=503)
        src, _ = slam.map_png_bytes()
        if src is None:
            return Response(b'no live map', status=204)
    img = cv2.imdecode(np.frombuffer(src, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return Response(b'decode failed', status=500)
    try:
        open_k = int(request.args.get('declutter', 13))
    except ValueError:
        open_k = 13
    open_k = max(3, min(41, open_k))
    if open_k % 2 == 0:
        open_k += 1
    try:
        eps = int(request.args.get('straighten', 18))
    except ValueError:
        eps = 18
    eps = max(2, min(80, eps)) / 1000.0
    out = _clean_map_img(img, cv2, np, open_k, eps)
    ok, buf = cv2.imencode('.png', out)
    if not ok:
        return Response(b'encode failed', status=500)
    cleaned = buf.tobytes()
    save = (request.args.get('save') or '').strip()
    if save:
        ss = ''.join(c for c in save if c.isalnum() or c in '-_')
        if ss:
            import shutil
            try:
                with open(os.path.join(SLAM_MAP_DIR, ss + '.png'), 'wb') as f:
                    f.write(cleaned)
                # If cleaning a SAVED map, copy its SLAM sidecars so the cleaned
                # output is a full, selectable map (same navigable data, clean
                # preview) rather than a loose PNG — list_maps keys off .posegraph.
                if name:
                    src_safe = ''.join(c for c in name if c.isalnum() or c in '-_')
                    for ext in ('.posegraph', '.data', '.meta.json'):
                        srcf = os.path.join(SLAM_MAP_DIR, src_safe + ext)
                        if os.path.isfile(srcf):
                            shutil.copy(srcf, os.path.join(SLAM_MAP_DIR, ss + ext))
            except OSError:
                pass
    resp = Response(cleaned, mimetype='image/png')
    resp.headers['Cache-Control'] = 'no-store'
    return resp


# ---------- Autonomous nav endpoints ------------------------------------------
@app.route('/auto/status')
def auto_status():
    if auto is None:
        return jsonify(available=False, state='unavailable',
                       error='auto_module not importable')
    s = auto.status()
    s['available'] = True
    return jsonify(s)


@app.route('/auto/goal', methods=['POST'])
def auto_goal():
    """Plan a path from current pose to (x, y) in map coords.

    Goal is consumed synchronously — A* runs inline and the response carries
    the planned path or the failure reason. The bot does NOT start driving
    yet; the user has to follow up with POST /auto/go."""
    if auto is None:
        return jsonify(ok=False, error='AutoNav unavailable'), 503
    data = request.get_json(silent=True) or {}
    try:
        x = float(data['x']); y = float(data['y'])
    except (KeyError, TypeError, ValueError):
        return jsonify(ok=False, error='need numeric x, y'), 400
    ok, msg = auto.set_goal(x, y)
    s = auto.status()
    return jsonify(ok=ok, msg=msg, **s)


@app.route('/auto/go', methods=['POST'])
def auto_go():
    if auto is None:
        return jsonify(ok=False, error='AutoNav unavailable'), 503
    if driver.is_killed():
        return jsonify(ok=False, error='driver is killed — rearm first'), 400
    ok, msg = auto.go()
    return jsonify(ok=ok, msg=msg)


@app.route('/auto/stop', methods=['POST'])
def auto_stop():
    if auto is None:
        return jsonify(ok=False, error='AutoNav unavailable'), 503
    ok, msg = auto.stop()
    return jsonify(ok=ok, msg=msg)


@app.route('/auto/clear', methods=['POST'])
def auto_clear():
    if auto is None:
        return jsonify(ok=False, error='AutoNav unavailable'), 503
    ok, msg = auto.clear()
    return jsonify(ok=ok, msg=msg)


@app.route('/auto/steer_polarity', methods=['GET', 'POST'])
def auto_steer_polarity():
    """Read or set the auto-drive steer polarity.

    GET → {"polarity": +1 | -1}
    POST {"polarity": -1} to flip — use when SLAM rotation doesn't match
    physical rotation (likely IMU gyro Z mounted inverted)."""
    if auto is None:
        return jsonify(ok=False, error='AutoNav unavailable'), 503
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        p = auto.set_steer_polarity(data.get('polarity', 1))
        return jsonify(ok=True, polarity=p)
    return jsonify(polarity=auto.get_steer_polarity())


@app.route('/slam/map.png')
def slam_map_png():
    if slam is None:
        return Response(b'', status=503)
    png, meta = slam.map_png_bytes()
    if png is None:
        return Response(b'', status=204)  # No map yet
    resp = Response(png, mimetype='image/png')
    # Stash metadata in headers so the client can position the image correctly.
    if meta:
        resp.headers['X-Map-W'] = str(meta['w'])
        resp.headers['X-Map-H'] = str(meta['h'])
        resp.headers['X-Map-Res'] = str(meta['res'])
        resp.headers['X-Map-OX'] = str(meta['ox'])
        resp.headers['X-Map-OY'] = str(meta['oy'])
    resp.headers['Cache-Control'] = 'no-store'
    return resp


def main():
    global driver, camera, telemetry, lidar, slam, auto
    camera = Camera()  # auto-probes /dev/video0..4, quality preset from file
    telemetry = Telemetry()
    lidar = Lidar()
    driver = Driver()
    slam = (SlamManager(serial_lidar=lidar, driver=driver)
            if SlamManager is not None else None)
    auto = (AutoNav(slam, driver)
            if AutoNav is not None and slam is not None else None)
    if slam is not None:
        # Auto-stop: if SLAM loses track (large pose jumps), the bot may not
        # know where it is — best to halt rather than drive blind. Soft stop
        # uses the same 1s latch as the user's stop button. Also abort any
        # in-progress autonomous drive so it doesn't keep re-issuing commands.
        def _on_lost():
            print('SLAM lost track — auto soft-stop', flush=True)
            if auto is not None:
                try: auto.stop()
                except Exception: pass
            try: driver.stop()
            except Exception as e: print(f'auto-stop failed: {e}', flush=True)
        slam.set_lost_callback(_on_lost)
    print(f'PWM: pwm0 throttle (neutral {THROTTLE_NEUTRAL}, max {THROTTLE_FWD_MAX}), '
          f'pwm1 steer (mid {STEER_MID}, [{STEER_MIN},{STEER_MAX}])', flush=True)
    threading.Thread(target=_core_watchdog, daemon=True).start()

    def shutdown(*_):
        print('\nshutting down…')
        # Stop SLAM first so the ROS subprocess gets a clean signal and
        # /dev/ttyUSB0 is released before we tear down the serial Lidar.
        if slam is not None and slam.state in ('active', 'starting'):
            try: slam.stop()
            except Exception: pass
            for _ in range(20):
                if slam.state == 'idle':
                    break
                time.sleep(0.2)
        try: driver.shutdown()
        except Exception: pass
        try: camera.shutdown()
        except Exception: pass
        try: lidar.shutdown()
        except Exception: pass
        os._exit(0)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    port = int(os.environ.get('PORT', 8888))
    print(f'dashboard → http://0.0.0.0:{port}/  (Ctrl-C to stop)', flush=True)
    app.run(host='0.0.0.0', port=port, threaded=True, debug=False)


if __name__ == '__main__':
    main()
