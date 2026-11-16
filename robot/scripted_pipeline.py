#!/usr/bin/env python3
"""
MiRo-E Scripted pipeline (the Scripted / control condition of the Phase 2 study)

Control condition: driven by an internal state machine, independent of any sensed context.

Design principles (controlled against the Spontaneous condition):
  + same action library (12 primitives)
  + same number of triggers (fixed 15 s, matching the Spontaneous 15 s base interval)
  + same touch-sensor reactions (without the random post-processing)
  + same director (identical motor-control layer)
  - no camera in the decisions, no VLM
  - action selection: a fixed-order pointer, independent of context
  - energy curve: a time-driven step state machine, independent of the user's posture

Also:
  - touch wake-up: touch in DROWSY -> c3_sudden_wake -> back to the LAZY stage
  - participant profile loaded from tuner_profile.json (shown in the UI only)
  - camera frames and UI state are published for experiment_ui.py

State steps:
  ALERT  (0-2 min)    -> curious and active, large actions
  CALM   (2-4 min)    -> calm observing, medium actions
  LAZY   (4-6 min)    -> lazy and bored, small actions
  DROWSY (6 min+)     -> drowsy, sleep chain
"""

import rospy
import json
import time
import threading
import math
import os
import cv2
import numpy as np
from std_msgs.msg import String, UInt16
from sensor_msgs.msg import CompressedImage

PROFILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "tuner_profile.json")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TRIGGER_INTERVAL = 15.0   # fixed 15 s, matching the Spontaneous base interval

# time thresholds of the energy states (s); a whole session lasts about 8 minutes
STATE_THRESHOLDS = {
    "ALERT":  (0,       2 * 60),   # 0-2 min: alert
    "CALM":   (2 * 60,  4 * 60),   # 2-4 min: calm observing
    "LAZY":   (4 * 60,  6 * 60),   # 4-6 min: lazy, bored
    "DROWSY": (6 * 60,  float("inf")),  # 6-8 min+: drowsy
}

# fixed action sequence per state (fixed order, pointer advances and wraps)
ACTION_POOLS = {
    "ALERT":  [
        "b1_curious_tilt",
        "b5_attention_shift",
        "b6_happy_wobble",
        "b2_ambient_scan",
        "b4_double_nod",
    ],
    "CALM": [
        "a7_curious_side",
        "a8_wide_scan",
        "b3_soft_stretch",
        "a3_neck_roll",
        "b4_double_nod",
    ],
    "LAZY": [
        "a6_slow_stretch_blink",
        "a3_neck_roll",
        "a7_curious_side",
        "b2_ambient_scan",
    ],
    "DROWSY": [
        "c1_getting_drowsy",
        "c2_sleep_snore",
        "c3_sudden_wake",
        "a6_slow_stretch_blink",
    ],
}

# fixed intensity per state
STATE_INTENSITY = {
    "ALERT":  0.85,
    "CALM":   0.70,
    "LAZY":   0.55,
    "DROWSY": 0.40,
}

# LED pattern (fixed, not driven by any mood)
STATE_LED = {
    "ALERT":  "focused",
    "CALM":   "normal",
    "LAZY":   "normal",
    "DROWSY": "normal",
}

# touch response: two fixed steps (no random aftermath)
TOUCH_STEP1_ACTION    = "petting_reaction"
TOUCH_STEP1_DURATION  = 4.0    # s
TOUCH_STEP2_ACTION    = "b4_double_nod"   # fixed closing action (not random)
TOUCH_STEP2_DURATION  = 3.0

# debounce parameters
TOUCH_CONFIRM_DUR  = 0.6
TOUCH_LOSE_DUR     = 0.5
TOUCH_COOLDOWN     = 12.0

C_STATE = '\033[96m'    # cyan: state information
C_ACT   = '\033[92m'    # green: action execution
C_TOUCH = '\033[93m'    # yellow: touch events
C_RES   = '\033[0m'


# ─────────────────────────────────────────────
# participant profile (for the UI display only)
# ─────────────────────────────────────────────
def load_profile():
    default = {
        "participantId":  "—",
        "profileType":    "—",
        "mentalMetaphor": "",
        "lma_calm":       {},
        "lma_active":     {},
    }
    try:
        with open(PROFILE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        for k in default:
            if k in data:
                default[k] = data[k]
        print(f"[Profile] Loaded: {default['participantId']}  {default['profileType']}")
    except Exception as e:
        print(f"[Profile] Load failed ({e}), using empty default.")
    return default


PROFILE = load_profile()


# ─────────────────────────────────────────────
# touch debounce filter (same logic as the Spontaneous pipeline)
# ─────────────────────────────────────────────
class SignalFilter:
    def __init__(self, confirm_dur, lose_dur, cooldown_dur):
        self.state       = False
        self._active_t   = None
        self._inactive_t = None
        self._last_rise  = 0.0
        self.confirm     = confirm_dur
        self.lose        = lose_dur
        self.cooldown    = cooldown_dur

    def update(self, raw):
        now = time.time()
        event = None
        if raw:
            self._inactive_t = None
            if not self.state:
                if self._active_t is None:
                    self._active_t = now
                elif (now - self._active_t >= self.confirm
                      and now - self._last_rise >= self.cooldown):
                    self.state = True
                    event = "RISING"
                    self._last_rise = now
                    self._active_t  = now
        else:
            self._active_t = None
            if self.state:
                if self._inactive_t is None:
                    self._inactive_t = now
                elif now - self._inactive_t >= self.lose:
                    self.state = False
                    event = "FALLING"
        return self.state, event


# ─────────────────────────────────────────────
# touch sensor (head only; body bit 0 connected)
# ─────────────────────────────────────────────
class TouchSensor:
    def __init__(self):
        self.head_touched = False
        self.body_touched = False
        rospy.Subscriber("/miro/sensors/touch_head", UInt16, self._cb_head, queue_size=1)
        rospy.Subscriber("/miro/sensors/touch_body", UInt16, self._cb_body, queue_size=1)

    def _cb_head(self, msg):
        valid = msg.data & 0x003F
        self.head_touched = bin(valid).count('1') >= 2

    def _cb_body(self, msg):
        valid = msg.data & 0x0001
        self.body_touched = (valid > 0)


# ─────────────────────────────────────────────
# state machine
# ─────────────────────────────────────────────
class StateMachine:
    def __init__(self):
        self._session_start = time.time()
        self._pointers      = {s: 0 for s in ACTION_POOLS}

    def get_state(self):
        elapsed = time.time() - self._session_start
        for state, (lo, hi) in STATE_THRESHOLDS.items():
            if lo <= elapsed < hi:
                return state
        return "DROWSY"

    def elapsed_str(self):
        e = int(time.time() - self._session_start)
        return f"{e//60}m{e%60:02d}s"

    def next_action(self):
        """Next action of the current state (advances in order and wraps around)."""
        state = self.get_state()
        pool  = ACTION_POOLS[state]
        idx   = self._pointers[state] % len(pool)
        action = pool[idx]
        self._pointers[state] = (idx + 1) % len(pool)
        return action, STATE_INTENSITY[state], STATE_LED[state], state

    def soft_reset_to_lazy(self):
        """Called after a touch wake-up: moves the session clock back to just after the start of LAZY,
        so the state machine goes through LAZY -> DROWSY again (falls asleep again after about 2 minutes)"""
        lazy_lo, _ = STATE_THRESHOLDS["LAZY"]
        # land 30 s after the start of LAZY, as a small buffer
        self._session_start = time.time() - (lazy_lo + 30)
        # also reset the DROWSY pointer so falling asleep again starts at c1
        self._pointers["DROWSY"] = 0


# ─────────────────────────────────────────────
# publish an action
# ─────────────────────────────────────────────
def publish(pub, action, intensity, led_mode):
    pub.publish(json.dumps({
        "action":    action,
        "intensity": intensity,
        "led_mode":  led_mode,
    }))


# ─────────────────────────────────────────────
# touch handling thread
# ─────────────────────────────────────────────
def touch_handler(pub, touch_lock, sm):
    """
    Touch response depends on the current state:
      - DROWSY: c3_sudden_wake -> b4_double_nod -> back to the LAZY stage (drowsy again later)
      - other states: petting_reaction -> b4_double_nod (original behaviour)
    """
    state = sm.get_state()

    if state == "DROWSY":
        print(f"\n{C_TOUCH}[touch] woken from sleep -> c3_sudden_wake{C_RES}")
        publish(pub, "c3_sudden_wake", 0.90, "normal")
        time.sleep(3.0)
        print(f"{C_TOUCH}[touch] wake-up ending -> b4_double_nod{C_RES}")
        publish(pub, "b4_double_nod", 0.75, "happy")
        time.sleep(2.0)
        sm.soft_reset_to_lazy()
        print(f"{C_TOUCH}[touch] back to the LAZY stage; asleep again in about 2 minutes{C_RES}")
    else:
        print(f"\n{C_TOUCH}[touch] start: {TOUCH_STEP1_ACTION}{C_RES}")
        publish(pub, TOUCH_STEP1_ACTION, 0.85, "petting")
        time.sleep(TOUCH_STEP1_DURATION)
        print(f"{C_TOUCH}[touch] ending: {TOUCH_STEP2_ACTION}{C_RES}")
        publish(pub, TOUCH_STEP2_ACTION, 0.75, "happy")
        time.sleep(TOUCH_STEP2_DURATION)

    print(f"{C_TOUCH}[touch] done, back to the main loop{C_RES}")
    touch_lock.clear()


# ─────────────────────────────────────────────
# camera bridge (display only, not used for decisions)
# ─────────────────────────────────────────────
class MiroCameraBridge:
    """Subscribe to MiRo's right eye and forward it to the UI. No recognition."""
    def __init__(self, pub_eye):
        self.pub_eye = pub_eye
        rospy.Subscriber("/miro/sensors/camr/compressed", CompressedImage,
                         self._cb, queue_size=1)
    def _cb(self, msg):
        try:
            self.pub_eye.publish(msg)  # forwarded unchanged
        except Exception:
            pass


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    rospy.init_node('miro_mechanical_pipeline', anonymous=True)
    pub          = rospy.Publisher('/miro/behavior_cmd',  String,          queue_size=1)
    pub_ui       = rospy.Publisher('/miro/ui_data',       String,          queue_size=1)
    pub_frame_ext= rospy.Publisher('/miro/ui_frame_ext',  CompressedImage, queue_size=1)
    pub_frame_eye= rospy.Publisher('/miro/ui_frame_eye',  CompressedImage, queue_size=1)

    sm     = StateMachine()
    touch  = TouchSensor()
    filt   = SignalFilter(TOUCH_CONFIRM_DUR, TOUCH_LOSE_DUR, TOUCH_COOLDOWN)
    cam_br = MiroCameraBridge(pub_frame_eye)

    # USB camera: only sent to the UI for display (no posture recognition)
    usb_cap = cv2.VideoCapture(0)

    touch_lock = threading.Event()
    rospy.sleep(1.0)

    print(f"\n{'='*52}")
    print(f"  🤖 MiRo Mechanical Pipeline (integrated)")
    print(f"  Participant : {PROFILE.get('participantId','?')}")
    print(f"  Profile     : {PROFILE.get('profileType','?')}")
    print(f"  trigger interval : {TRIGGER_INTERVAL:.0f}s (fixed)")
    print(f"  state steps      : ALERT(0-2m) -> CALM(2-4m) -> LAZY(4-6m) -> DROWSY(6m+)")
    print(f"  touch wake-up    : touch in DROWSY -> c3_sudden_wake -> back to LAZY")
    print(f"{'='*52}\n")

    # publish an initial action at start-up
    action, intensity, led, state = sm.next_action()
    publish(pub, action, intensity, led)
    print(f"{C_STATE}[{sm.elapsed_str()}] initial state: {state}{C_RES}")
    print(f"{C_ACT}  → {action.upper()}  intensity={intensity:.2f}{C_RES}\n")

    last_trigger = time.time()
    last_ui_pub  = 0
    rate = rospy.Rate(20)

    while not rospy.is_shutdown():
        now = time.time()

        # -- touch detection -------------------------------------------------
        raw_touch = touch.head_touched or touch.body_touched
        _, t_event = filt.update(raw_touch)

        if t_event == "RISING" and not touch_lock.is_set():
            touch_lock.set()
            last_trigger = now
            threading.Thread(
                target=touch_handler,
                args=(pub, touch_lock, sm),
                daemon=True
            ).start()

        # -- timed trigger (skipped while a touch is being handled) ----------
        if not touch_lock.is_set() and now - last_trigger >= TRIGGER_INTERVAL:
            last_trigger = now
            action, intensity, led, state = sm.next_action()
            publish(pub, action, intensity, led)
            print(f"{C_STATE}[{sm.elapsed_str()}] state: {state}{C_RES}"
                  f"  ptr={sm._pointers[state]-1}")
            print(f"{C_ACT}  → {action.upper()}  intensity={intensity:.2f}{C_RES}\n")

        # -- forward the USB camera to the UI (display only) ----------------
        if usb_cap.isOpened():
            ret, usb_frame = usb_cap.read()
            if ret:
                usb_frame = cv2.flip(usb_frame, 1)
                try:
                    _, jpg = cv2.imencode(".jpg", usb_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    msg = CompressedImage()
                    msg.format = "jpeg"
                    msg.data   = jpg.tobytes()
                    pub_frame_ext.publish(msg)
                except Exception:
                    pass

        # -- publish the UI state now and then (10 Hz is enough) ------------
        if now - last_ui_pub > 0.1:
            last_ui_pub = now
            ui = {
                "mode":          "mechanical",
                "participant":   PROFILE.get("participantId", "—"),
                "profile_type":  PROFILE.get("profileType",  "—"),
                "metaphor":      PROFILE.get("mentalMetaphor", ""),
                "lma_calm":      PROFILE.get("lma_calm",   {}),
                "lma_active":    PROFILE.get("lma_active", {}),
                "current_action":action,
                "elapsed":       sm.elapsed_str(),
                "state":         sm.get_state(),
                "is_petted":     touch_lock.is_set(),
                "is_looking":    False,
                "visual":        "",
                "thought":       "",
                "reason":        f"Scripted sequence — state {sm.get_state()}",
            }
            try:
                pub_ui.publish(json.dumps(ui, ensure_ascii=False))
            except Exception:
                pass

        rate.sleep()

    usb_cap.release()


if __name__ == "__main__":
    main()
