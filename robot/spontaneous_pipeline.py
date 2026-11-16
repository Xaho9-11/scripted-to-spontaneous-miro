"""
MiRo-E Spontaneous pipeline (the Spontaneous condition of the Phase 2 study)
Built on the stand-alone VLM pipeline, adding:
  - the participant profile is loaded from tuner_profile.json (written by profile_loader.py)
  - UI state is published on /miro/ui_data for experiment_ui.py
  - kept: posture calibration, MediaPipe, touch reflex, three-stage recovery, no-idle prompt
"""

import cv2
import mediapipe as mp
import json, time, copy, threading, os
import numpy as np
import random
from collections import deque
from std_msgs.msg import String, UInt16
from sensor_msgs.msg import CompressedImage
import rospy

SHOW_WINDOW = False   # video is shown by experiment_ui.py
VLM_MODE = "cloud_gemini"
LOCAL_MODEL_NAME = "qwen3.5:2b"
CLOUD_MODEL_NAME = "gemini-3.1-flash-lite"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # set in the environment; never commit a key

PROFILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "tuner_profile.json")

# ─────────────────────────────────────────────
# [Fix F] trigger interval
# ─────────────────────────────────────────────
TRIGGER_INTERVAL_BASE = 15.0   # the VLM is called every 15 s on average
TRIGGER_JITTER        = 3.0    # +/- 3 s random jitter, i.e. 12-18 s

# ─────────────────────────────────────────────
# Allowed actions (only those with a clearly visible effect when chosen by the VLM)
# micro-movements (a1/a2/a4/a5) live in the director's biological-rhythm layer; the VLM no longer calls them
# c2_sleep_snore is started by the director's timer, never chosen by the VLM directly
# ─────────────────────────────────────────────
VALID_ATOMS = [
    # A series: clearly visible movements
    "a3_neck_roll",          # slow neck circle
    "a6_slow_stretch_blink", # head up, eyes close and reopen
    "a7_curious_side",       # pronounced head tilt to one side
    "a8_wide_scan",          # 180-degree look-around
    # B series: active interaction
    "b1_curious_tilt",       # curious tilt
    "b2_ambient_scan",       # scan the surroundings
    "b3_soft_stretch",       # stretch
    "b4_double_nod",         # double nod
    "b5_attention_shift",    # quick head turn
    "b6_happy_wobble",       # happy wobble
    # C series: sleep chain (c1 and c3 only; c2 is timer-driven)
    "c1_getting_drowsy",     # getting drowsy
    "c3_sudden_wake",        # sudden wake
]

# fallback candidates: only actions with a clearly visible motion
A_SERIES_FALLBACK = ["a3_neck_roll", "a7_curious_side", "a8_wide_scan",
                     "a6_slow_stretch_blink", "b2_ambient_scan", "b4_double_nod"]

# ─────────────────────────────────────────────
# Participant profile (loaded from tuner_profile.json)
# ─────────────────────────────────────────────
def load_participant_profile():
    """Load tuner_profile.json; fall back to the default Pattern Breaker profile on failure."""
    default = {
        "participantId": "—",
        "profileType":   "Pattern Breaker",
        "mentalMetaphor": "",
        "calm_preference": (
            "Human is focused. Show life-like presence with VISIBLE movements. "
            "Prefer A-visible (a3/a6/a7/a8) or calm B-series (b2/b3/b4) with intensity 0.6-0.8. "
            "Even a focused human deserves a robot companion that clearly moves around."
        ),
        "active_preference": (
            "Human seems stressed or deeply focused. Use A-visible or B-series. "
            "Do not choose micro-movements — they look broken. "
            "Pick actions with clear neck/head movement (a3/a7/a8/b2/b4) intensity 0.6-0.8."
        ),
        "lma_calm":   {"weight": 45, "time": 54, "flow": 82, "space": 28},
        "lma_active": {"weight": 75, "time": 64, "flow": 75, "space": 20},
    }
    try:
        with open(PROFILE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        # merge: values present in the JSON override the defaults
        for k in default:
            if k in data:
                default[k] = data[k]
        # accept the older keys (calm_pref / active_pref)
        if "calm_pref"   in data: default["calm_preference"]   = data["calm_pref"]
        if "active_pref" in data: default["active_preference"] = data["active_pref"]
        print(f"[Profile] Loaded: {default['participantId']}  {default['profileType']}")
    except Exception as e:
        print(f"[Profile] Load failed ({e}), using default Pattern Breaker.")
    return default

PARTICIPANT_PROFILE = load_participant_profile()

# ─────────────────────────────────────────────
# Global session state
# ─────────────────────────────────────────────
SESSION_START_TIME = None
ACTION_HISTORY     = deque(maxlen=6)

shared_miro_frame = None
shared_summary    = None
is_thinking       = False

current_vlm_action = "a4_focus_mirror"
current_vlm_led    = "normal"

# UI state (updated by the VLM worker, published by main on /miro/ui_data)
ui_state = {
    "mode":          "spontaneous",
    "participant":   PARTICIPANT_PROFILE.get("participantId", "—"),
    "profile_type":  PARTICIPANT_PROFILE.get("profileType",  "—"),
    "metaphor":      PARTICIPANT_PROFILE.get("mentalMetaphor", ""),
    "lma_calm":      PARTICIPANT_PROFILE.get("lma_calm",   {}),
    "lma_active":    PARTICIPANT_PROFILE.get("lma_active", {}),
    "current_action":"a4_focus_mirror",
    "posture":       "—",
    "is_petted":     False,
    "is_looking":    False,
    "visual":        "",
    "thought":       "",
    "reason":        "",
}

# timer handles of the touch-recovery stages (cancelled when a new touch arrives)
recovery_timers = []

C_VIS  = '\033[96m'
C_THO  = '\033[95m'
C_ACT  = '\033[92m'
C_WARN = '\033[91m'
C_RES  = '\033[0m'

# ─────────────────────────────────────────────
# Debounce filter
# ─────────────────────────────────────────────
class SignalFilter:
    def __init__(self, confirm_duration, lose_duration, cooldown_duration=1.5):
        self.state = False
        self.active_start   = None
        self.inactive_start = None
        self.last_rising_time = 0.0
        self.confirm_dur  = confirm_duration
        self.lose_dur     = lose_duration
        self.cooldown_dur = cooldown_duration

    def update(self, raw_signal):
        now = time.time()
        event = None
        if raw_signal:
            self.inactive_start = None
            if not self.state:
                if self.active_start is None:
                    self.active_start = now
                elif now - self.active_start >= self.confirm_dur:
                    if now - self.last_rising_time >= self.cooldown_dur:
                        self.state = True
                        event = "RISING"
                        self.last_rising_time = now
                    else:
                        self.active_start = now
        else:
            self.active_start = None
            if self.state:
                if self.inactive_start is None:
                    self.inactive_start = now
                elif now - self.inactive_start >= self.lose_dur:
                    self.state = False
                    event = "FALLING"
        return self.state, event


touch_filter = SignalFilter(confirm_duration=0.6, lose_duration=0.5, cooldown_duration=12.0)
gaze_filter  = SignalFilter(confirm_duration=0.6, lose_duration=0.8, cooldown_duration=3.0)

global_is_petted = False
global_is_looking = False
latch_petted_for_vlm  = False
latch_looking_for_vlm = False
reflex_petted_start = False
reflex_petted_end   = False
reflex_gaze_start   = False
reflex_gaze_end     = False

# ─────────────────────────────────────────────
# Camera and touch sensors
# ─────────────────────────────────────────────
class MiroSensoryNode:
    def __init__(self):
        self.frame_r = None
        self.head_touched = False
        self.body_touched = False
        base = "/miro"
        rospy.Subscriber(f"{base}/sensors/camr/compressed", CompressedImage, self._cb_r, queue_size=1)
        rospy.Subscriber(f"{base}/sensors/touch_head", UInt16, self._cb_touch_head, queue_size=1)
        rospy.Subscriber(f"{base}/sensors/touch_body", UInt16, self._cb_touch_body, queue_size=1)

    def _cb_r(self, msg):
        try:
            self.frame_r = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        except: pass

    def _cb_touch_head(self, msg):
        valid_bits = msg.data & 0x003F
        self.head_touched = bin(valid_bits).count('1') >= 2

    def _cb_touch_body(self, msg):
        valid_bits = msg.data & 0x0001
        self.body_touched = valid_bits > 0

    def read_frame(self):
        if self.frame_r is not None:
            h, w = self.frame_r.shape[:2]
            cropped = self.frame_r[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
            return True, cv2.resize(cropped, (480, 360))
        return False, None


class EgocentricFaceDetector:
    def __init__(self):
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(min_detection_confidence=0.5, model_selection=0)

    def check_gaze(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.detector.process(rgb)
        raw_looking = False
        if res.detections:
            det = res.detections[0]
            kp = det.location_data.relative_keypoints
            r_eye_x, l_eye_x, nose_x = kp[0].x, kp[1].x, kp[2].x
            eye_dist = abs(l_eye_x - r_eye_x)
            if eye_dist > 0.05:
                mid_eye = (l_eye_x + r_eye_x) / 2.0
                symmetry = abs(mid_eye - nose_x) / eye_dist
                if symmetry < 0.18:
                    raw_looking = True
                    bbox = det.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
                    cv2.rectangle(frame, (x1, y1),
                                  (x1+int(bbox.width*w), y1+int(bbox.height*h)),
                                  (0, 0, 255), 2)
                    cv2.putText(frame, "GAZE DETECTED", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return raw_looking, frame

# ─────────────────────────────────────────────
# MediaPipe posture extraction
# ─────────────────────────────────────────────
class UserState:
    def __init__(self, posture, n2s=0.0, nz=0.0, hdrop=0.0):
        self.posture = posture
        self.nose_to_shoulder = n2s
        self.nose_z_offset    = nz
        self.head_drop        = hdrop


class PoseExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose    = self.mp_pose.Pose(min_detection_confidence=0.6,
                                         min_tracking_confidence=0.6)
        self._history      = []
        self._history_size = 8
        # baseline: replaced after calibration; conservative defaults before it
        self.baseline_z     = 0.797
        self.baseline_n2s   = 0.311
        self.baseline_hdrop = 0.025
        self._calibrated    = False

    def calibrate(self, cap, duration_sec=5):
        """
        Ask the participant to sit upright and record a baseline for duration_sec seconds.
        After calibration, posture is judged against this personal baseline, which is much more accurate.
        """
        print(f"\n{'='*50}")
        print(f"  📐 posture calibration started")
        print(f"  please sit upright and keep still for {duration_sec} s...")
        print(f"{'='*50}")

        sz, sn, sh = [], [], []
        start = time.time()
        P = self.mp_pose.PoseLandmark

        while time.time() - start < duration_sec:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            res   = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if res.pose_landmarks:
                lm    = res.pose_landmarks.landmark
                nose  = lm[P.NOSE]
                l_sh  = lm[P.LEFT_SHOULDER];  r_sh  = lm[P.RIGHT_SHOULDER]
                l_ear = lm[P.LEFT_EAR];       r_ear = lm[P.RIGHT_EAR]

                sz.append((l_sh.z + r_sh.z) / 2 - nose.z)
                sn.append((l_sh.y + r_sh.y) / 2 - nose.y)
                sh.append(nose.y - (l_ear.y + r_ear.y) / 2)

                # calibration progress bar
                elapsed  = time.time() - start
                progress = int(elapsed / duration_sec * 30)
                bar      = '█' * progress + '░' * (30 - progress)
                print(f"\r  [{bar}] {elapsed:.1f}/{duration_sec}s  "
                      f"samples={len(sz)}", end='', flush=True)

            cv2.waitKey(1)

        print()   # new line

        if len(sz) < 10:
            print(f"  ⚠️  too few samples ({len(sz)} frames), using the default baseline")
            return False

        self.baseline_z     = sum(sz) / len(sz)
        self.baseline_n2s   = sum(sn) / len(sn)
        self.baseline_hdrop = sum(sh) / len(sh)
        self._calibrated    = True

        print(f"\n  ✅ calibration done ({len(sz)} frames)")
        print(f"     baseline_z    = {self.baseline_z:.4f}")
        print(f"     baseline_n2s  = {self.baseline_n2s:.4f}")
        print(f"     baseline_hdrop= {self.baseline_hdrop:.4f}")
        print(f"{'='*50}\n")
        return True

    def extract(self, frame):
        res = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            return None, frame

        lm    = res.pose_landmarks.landmark
        P     = self.mp_pose.PoseLandmark
        nose  = lm[P.NOSE]
        l_sh  = lm[P.LEFT_SHOULDER];  r_sh  = lm[P.RIGHT_SHOULDER]
        l_ear = lm[P.LEFT_EAR];       r_ear = lm[P.RIGHT_EAR]

        n2s   = (l_sh.y + r_sh.y) / 2 - nose.y
        nz    = (l_sh.z + r_sh.z) / 2 - nose.z
        hdrop = nose.y - (l_ear.y + r_ear.y) / 2

        # deviation from the personal baseline
        if   n2s < self.baseline_n2s - 0.09 or hdrop > self.baseline_hdrop + 0.08:
            raw = "head_down"
        elif nz  > self.baseline_z   + 0.13:
            raw = "forward_lean"
        elif nz  < self.baseline_z   - 0.06:
            raw = "lean_back"
        else:
            raw = "upright"

        self._history.append(raw)
        if len(self._history) > self._history_size:
            self._history.pop(0)
        posture = max(set(self._history), key=self._history.count)

        self.mp_draw.draw_landmarks(frame, res.pose_landmarks,
                                    self.mp_pose.POSE_CONNECTIONS)
        return UserState(posture, n2s, nz, hdrop), frame


class PostureBuffer:
    def __init__(self):
        self._last_sample = 0.0
        self._buffer = deque()

    def update(self, state):
        now = time.time()
        if now - self._last_sample < 1.0: return
        self._last_sample = now
        self._buffer.append((now, state.posture))
        cutoff = now - 30.0
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

    def summary(self):
        counts = {}
        for _, l in self._buffer: counts[l] = counts.get(l, 0) + 1
        return {"dominant": max(counts, key=counts.get) if counts else "unknown"}

    def history_str(self):
        if not self._buffer: return "no data"
        recent = list(self._buffer)[-8:]
        return " → ".join([p for _, p in recent])


class TriggerController:
    def __init__(self):
        self.last = time.time()
        self.next_interval = self._gen_interval()

    @staticmethod
    def _gen_interval():
        return TRIGGER_INTERVAL_BASE + random.uniform(-TRIGGER_JITTER, TRIGGER_JITTER)

    def check(self):
        if time.time() - self.last > self.next_interval:
            self.last = time.time()
            self.next_interval = self._gen_interval()
            return True
        return False

# ─────────────────────────────────────────────
# VLM client: the prompt rules out idle completely
# ─────────────────────────────────────────────
class VLMClient:
    def __init__(self, mode=VLM_MODE):
        self.mode = mode
        if mode == "cloud_gemini":
            from google import genai
            self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        elif mode == "cloud_openai":
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    # -- prompt ----------------------------------------------------------
    def _build_prompt(self, posture, is_petted, is_looking,
                      history_str, session_elapsed, current_act):
        touch_status = "YES (active touch on robot)" if is_petted else "NO"
        gaze_status  = "YES (human looking at robot)" if is_looking else "NO"

        # current recommendation
        if posture in ("forward_lean", "head_down"):
            profile_pref = PARTICIPANT_PROFILE["active_preference"]
        else:
            profile_pref = PARTICIPANT_PROFILE["calm_preference"]

        # look at recent history to build a diversity hint
        recent = [a for _, a, _ in list(ACTION_HISTORY)[-4:]]
        history_actions = ", ".join(recent) if recent else "none yet"

        # count recent A/B choices for the DIVERSITY RULE
        recent_a = sum(1 for a in recent if a.startswith("a"))
        recent_b = sum(1 for a in recent if a.startswith("b"))
        if recent_a >= 2 and recent_b == 0:
            diversity_hint = "⚠️ DIVERSITY: your last actions were all A-series. You MUST pick a B-series action now."
            recommended = "B series — b1/b2/b3/b4/b5/b6, intensity 0.7-0.9"
        elif recent_b >= 3:
            diversity_hint = "You've been active. A calmer visible A-series is fine now."
            recommended = "A-visible — a3/a6/a7/a8, intensity 0.6-0.8"
        else:
            diversity_hint = "Mix freely between A-visible and B-series."
            recommended = "A-visible or B-series, intensity 0.6-0.9"

        return f"""You are MiRo, an emotional companion robot dog observing your human.

═══ ACTION VOCABULARY (12 atoms — choose EXACTLY ONE) ═══

IMPORTANT: Micro-movements (breathing, weight shift, ear flicks) are handled
automatically by the robot's biological rhythm system. Do NOT pick them here.
Your choices must produce VISIBLE head/neck movements that a human can notice.

A-VISIBLE — calm but clearly visible (intensity 0.6-0.8):
  a3_neck_roll           — slow 8s circular neck movement
  a6_slow_stretch_blink  — tilt head back slowly, close eyes, then snap forward
  a7_curious_side        — sharp tilt to one side, hold 3s, return
  a8_wide_scan           — slow 180° horizontal head sweep (~8s)

B-ACTIVE — engaged, curious or joyful (intensity 0.7-0.9):
  b1_curious_tilt        — strong head tilt + both ears up
  b2_ambient_scan        — wide quick environment sweep
  b3_soft_stretch        — neck fully extends then returns
  b4_double_nod          — two clear downward nods
  b5_attention_shift     — fast head snap to the side
  b6_happy_wobble        — large left-right head wobble + high tail

C-SLEEP — only when ignored for a long time:
  c1_getting_drowsy      — eyes close, head droops (do NOT pick c2 — auto-managed)
  c3_sudden_wake         — startle wake, only if dramatic posture change after sleep

═══ SELECTION RULES ═══

1. NO idle, NO micro-movements (a1/a2/a4/a5 don't exist here).
2. {diversity_hint}
3. Do NOT repeat last action: '{current_act}'
4. A-visible and B-series are EQUAL in priority — do not always default to A.
   Even a focused human benefits from seeing the robot do something visible.
5. Sleep guidance (c1) only when: session > 3min AND no touch AND no gaze AND
   recent actions have been a7/a8/a3 (interest fading pattern).
6. c3_sudden_wake only if robot was in c1/c2 AND something dramatic changed.

═══ CONTEXT ═══

- Vision: [attached image — your right eye view]
- Human posture: '{posture}'
- Touch sensor: {touch_status}
- Eye contact: {gaze_status}
- Posture history (30s): {history_str}
- Your recent actions: {history_actions}  (A count: {recent_a}, B count: {recent_b})
- Session elapsed: {session_elapsed:.0f}s
- Profile: {profile_pref}
- Recommended: {recommended}

═══ OUTPUT — JSON only, no markdown ═══

{{
  "visual_zh": "<what you see, 1 sentence>",
  "thought_zh": "<your feeling, 1 sentence>",
  "decision_reason": "<why this action, 1 sentence>",
  "action": "<one of the 12 atoms above>",
  "intensity": 0.75,
  "duration": 6.0,
  "led_mode": "<normal | happy | focused | petting>"
}}"""

    # -- entry point -----------------------------------------------------
    def call(self, frame, posture, is_petted, is_looking,
             history_str, session_elapsed, current_act):
        prompt = self._build_prompt(posture, is_petted, is_looking,
                                    history_str, session_elapsed, current_act)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])

        for attempt in range(3):
            try:
                if self.mode == "cloud_gemini":
                    return self._call_gemini(buf.tobytes(), prompt)
                elif self.mode == "cloud_openai":
                    return self._call_openai(buf.tobytes(), prompt)
            except Exception as e:
                print(f"   ⚠️ VLM retry {attempt+1}/3: {e}")
                time.sleep(1.0)
        return {}

    def _call_gemini(self, img_bytes, prompt):
        from google.genai import types
        resp = self.gemini_client.models.generate_content(
            model=CLOUD_MODEL_NAME,
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.8     # slightly higher temperature for more varied actions
            )
        )
        return json.loads(resp.text.strip().replace("```json","").replace("```","").strip())

    def _call_openai(self, img_bytes, prompt):
        import base64
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        resp = self.openai_client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            temperature=0.8,
            messages=[{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
                {"type": "text", "text": prompt}
            ]}]
        )
        return json.loads(resp.choices[0].message.content)


def publish_action(action_name, led_mode, intensity, pub_cmd):
    pub_cmd.publish(json.dumps({
        "action":    action_name,
        "led_mode":  led_mode,
        "intensity": intensity
    }))

# ─────────────────────────────────────────────
# [Fix E] three-stage recovery after touch
# ─────────────────────────────────────────────
def cancel_recovery_timers():
    """Cancel all pending recovery phases when a new touch arrives."""
    global recovery_timers
    for t in recovery_timers:
        t.cancel()
    recovery_timers.clear()


def start_touch_recovery(pub_cmd):
    """
    Called when petting ends; starts a three-stage recovery:
    Phase 1 (now):    petting_aftermath  -> 3 s smooth ending
    Phase 2 (3.0 s):  b6_happy_wobble    -> 1.5 s transition
    Phase 3 (4.5 s):  back to VLM control
    """
    global recovery_timers
    cancel_recovery_timers()

    # Phase 1: go into the aftermath immediately
    publish_action("petting_aftermath", "happy", 0.7, pub_cmd)
    print(f"{C_VIS}☁️  [Phase 1/3] petting_aftermath (3 s aftermath){C_RES}")

    def phase2():
        if reflex_lock_state.get("locked", False):
            return  # a new touch happened, skip
        print(f"{C_VIS}☁️  [Phase 2/3] b6_happy_wobble (1.5 s transition){C_RES}")
        publish_action("b6_happy_wobble", "happy", 0.6, pub_cmd)

    def phase3():
        if reflex_lock_state.get("locked", False):
            return
        print(f"{C_VIS}☁️  [Phase 3/3] back to VLM control -> {current_vlm_action.upper()}{C_RES}")
        publish_action(current_vlm_action, current_vlm_led, 0.6, pub_cmd)

    t2 = threading.Timer(3.0, phase2)
    t3 = threading.Timer(4.5, phase3)
    recovery_timers.extend([t2, t3])
    t2.start()
    t3.start()


# reflex_lock wrapped in a dict so closures read the latest value
reflex_lock_state = {"locked": False}

# ─────────────────────────────────────────────
# VLM worker thread
# ─────────────────────────────────────────────
def vlm_worker_thread(vlm_instance, pub_cmd):
    global shared_miro_frame, shared_summary, is_thinking
    global latch_petted_for_vlm, latch_looking_for_vlm
    global current_vlm_action, current_vlm_led

    while not rospy.is_shutdown():
        if shared_miro_frame is not None and shared_summary is not None and not is_thinking:
            is_thinking = True
            m_frame     = copy.deepcopy(shared_miro_frame)
            dom_posture = shared_summary.get("dominant", "upright")
            history_str = shared_summary.get("history_str", "unknown")

            petted_snap  = global_is_petted  or latch_petted_for_vlm
            looking_snap = global_is_looking or latch_looking_for_vlm
            latch_petted_for_vlm  = False
            latch_looking_for_vlm = False

            shared_miro_frame = None
            shared_summary    = None

            session_elapsed = time.time() - (SESSION_START_TIME or time.time())

            res = vlm_instance.call(
                m_frame, dom_posture, petted_snap, looking_snap,
                history_str, session_elapsed, current_vlm_action
            )

            action    = res.get("action", "")
            intensity = float(res.get("intensity", 0.6))
            duration  = float(res.get("duration",  5.0))
            led_mode  = res.get("led_mode", "normal")

            # -- [Fix D] fallback: unexpected idle or unknown action -> force an A-series action
            if action not in VALID_ATOMS:
                if action.lower() == "idle":
                    print(f"   {C_WARN}⚠️ VLM returned idle, switching to an A-series action{C_RES}")
                else:
                    print(f"   {C_WARN}⚠️ unknown action '{action}', switching to an A-series action{C_RES}")
                # pick an A-series action different from the current one
                candidates = [a for a in A_SERIES_FALLBACK if a != current_vlm_action]
                action = random.choice(candidates or A_SERIES_FALLBACK)
                intensity = 0.55

            # keep durations within sensible bounds
            duration  = max(3.0, min(10.0, duration))
            intensity = max(0.3, min(1.0, intensity))

            print(f"\n{C_THO}┌── [💭 inner state] ──────────────────{C_RES}")
            print(f"{C_THO}│ visual:   {res.get('visual_zh','...')}{C_RES}")
            print(f"{C_THO}│ thought:  {res.get('thought_zh','...')}{C_RES}")
            print(f"{C_THO}│ reason:   {res.get('decision_reason','...')}{C_RES}")
            print(f"{C_ACT}└── action: {action.upper()}  "
                  f"intensity={intensity:.2f}  dur={duration:.1f}s  led={led_mode}{C_RES}")

            # update the UI state
            ui_state["visual"]         = res.get("visual_zh","")
            ui_state["thought"]        = res.get("thought_zh","")
            ui_state["reason"]         = res.get("decision_reason","")
            ui_state["current_action"] = action
            ui_state["posture"]        = dom_posture

            current_vlm_action = action
            current_vlm_led    = led_mode
            ACTION_HISTORY.append((time.time(), action, intensity))
            publish_action(action, led_mode, intensity, pub_cmd)
            time.sleep(duration)

            is_thinking = False

        time.sleep(0.05)


def main():
    global shared_miro_frame, shared_summary, is_thinking
    global reflex_petted_start, reflex_petted_end, reflex_gaze_start, reflex_gaze_end
    global global_is_looking, global_is_petted, latch_petted_for_vlm, latch_looking_for_vlm
    global current_vlm_action, current_vlm_led
    global SESSION_START_TIME

    rospy.init_node('miro_vlm_fusion_pipeline', anonymous=True)
    SESSION_START_TIME = time.time()
    pub_cmd = rospy.Publisher('/miro/behavior_cmd', String, queue_size=1)
    pub_ui  = rospy.Publisher('/miro/ui_data',     String, queue_size=1)
    pub_frame_ext = rospy.Publisher('/miro/ui_frame_ext', CompressedImage, queue_size=1)
    pub_frame_eye = rospy.Publisher('/miro/ui_frame_eye', CompressedImage, queue_size=1)

    USB_CAMERA_INDEX = 0
    usb_cap = cv2.VideoCapture(USB_CAMERA_INDEX)
    sensory = MiroSensoryNode()
    rospy.sleep(1.5)

    # [Fix G] publish an initial action at start-up so the robot does not stand still in default
    print(f"{C_ACT}🚀 initial action: a4_focus_mirror{C_RES}")
    publish_action("a4_focus_mirror", "normal", 0.6, pub_cmd)

    extractor = PoseExtractor()
    ego_face_detector = EgocentricFaceDetector()
    buf     = PostureBuffer()
    trigger = TriggerController()
    vlm     = VLMClient(mode=VLM_MODE)

    # -- posture calibration (required before each participant) ----------
    # camera warm-up
    print("[INFO] warming up the camera...")
    for _ in range(20):
        usb_cap.read()

    calibrated = extractor.calibrate(usb_cap, duration_sec=5)
    if not calibrated:
        print(f"{C_WARN}⚠️  calibration failed, continuing with the default baseline (posture may be judged less accurately){C_RES}")

    threading.Thread(target=vlm_worker_thread, args=(vlm, pub_cmd), daemon=True).start()

    print(f"\n{C_ACT}===================================================={C_RES}")
    print(f"{C_ACT} 🚀 Spontaneous pipeline started: calibration + no-idle prompt + three-stage touch recovery{C_RES}")
    print(f"{C_ACT}    trigger interval: {TRIGGER_INTERVAL_BASE}s +/- {TRIGGER_JITTER}s{C_RES}")
    print(f"{C_ACT}    posture calibration: {'done ✅' if calibrated else 'defaults ⚠️'}{C_RES}")
    print(f"{C_ACT}===================================================={C_RES}\n")

    while not rospy.is_shutdown():
        raw_touch = sensory.head_touched or sensory.body_touched
        t_state, t_event = touch_filter.update(raw_touch)
        global_is_petted = t_state

        if t_event == "RISING":
            reflex_petted_start = True
            latch_petted_for_vlm = True
        elif t_event == "FALLING":
            reflex_petted_end = True

        # -- touch starts: excited reaction; cancel any recovery in progress --
        if reflex_petted_start:
            cancel_recovery_timers()
            reflex_lock_state["locked"] = True
            print(f"\n{C_WARN}⚡ [reflex L1] petting started: excited{C_RES}")
            publish_action("petting_reaction", "petting", 0.8, pub_cmd)
            reflex_petted_start = False
            reflex_gaze_start = False

        # -- touch ends: start the three-stage recovery ----------------------
        if reflex_petted_end:
            reflex_lock_state["locked"] = False
            print(f"\n{C_VIS}🍃 [petting ended] starting the three-stage recovery{C_RES}")
            start_touch_recovery(pub_cmd)
            reflex_petted_end = False

        # -- gaze reflex -----------------------------------------------------
        if reflex_gaze_start:
            if not reflex_lock_state["locked"]:
                print(f"\n{C_VIS}👀 [reflex L2] eye contact: tilts the head and listens{C_RES}")
                publish_action("gaze_curious_reaction", "focused", 0.6, pub_cmd)
            reflex_gaze_start = False

        if reflex_gaze_end:
            if not reflex_lock_state["locked"]:
                print(f"\n☁️  [recovered] gaze moved away -> {current_vlm_action.upper()}{C_RES}")
                publish_action(current_vlm_action, current_vlm_led, 0.6, pub_cmd)
            reflex_gaze_end = False

        # -- visual processing -----------------------------------------------
        ret_usb, usb_frame = usb_cap.read()
        if not ret_usb:
            continue
        usb_frame = cv2.flip(usb_frame, 1)

        ret_m, m_frame = sensory.read_frame()
        state, ann_frame = extractor.extract(usb_frame)

        if state is not None and ret_m:
            buf.update(state)
            summary = buf.summary()
            dom = summary["dominant"]

            raw_looking, ann_m_frame = ego_face_detector.check_gaze(m_frame)
            g_state, g_event = gaze_filter.update(raw_looking)
            global_is_looking = g_state

            if g_event == "RISING":
                reflex_gaze_start = True
                latch_looking_for_vlm = True
            elif g_event == "FALLING":
                reflex_gaze_end = True

            # -- update the UI state live ---------------------------------------
            ui_state["posture"]    = state.posture
            ui_state["is_petted"]  = global_is_petted
            ui_state["is_looking"] = global_is_looking
            ui_state["calibrated"] = extractor._calibrated

            if trigger.check() and not is_thinking:
                shared_miro_frame = m_frame.copy()
                shared_summary = {**summary, "history_str": buf.history_str()}

            # -- send frames to the UI (always, whatever SHOW_WINDOW) ----------
            try:
                _, jpg_ext = cv2.imencode(".jpg", ann_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                msg_ext = CompressedImage()
                msg_ext.format = "jpeg"
                msg_ext.data   = jpg_ext.tobytes()
                pub_frame_ext.publish(msg_ext)

                _, jpg_eye = cv2.imencode(".jpg", ann_m_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                msg_eye = CompressedImage()
                msg_eye.format = "jpeg"
                msg_eye.data   = jpg_eye.tobytes()
                pub_frame_eye.publish(msg_eye)
            except Exception:
                pass

            # -- publish the UI status text --------------------------------------
            try:
                pub_ui.publish(json.dumps(ui_state, ensure_ascii=False))
            except Exception:
                pass

            if SHOW_WINDOW:
                cv2.putText(ann_frame, f"Posture: {state.posture}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
                cal_txt   = "CAL OK" if extractor._calibrated else "CAL DEFAULT"
                cal_color = (0, 255, 100) if extractor._calibrated else (0, 100, 255)
                cv2.putText(ann_frame, cal_txt,
                            (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cal_color, 1)
                status_color = (0, 0, 255) if global_is_petted else \
                               ((255, 0, 255) if global_is_looking else (0, 255, 0))
                status_text = "+++ PETTING! +++" if global_is_petted else \
                              ("!!! EYE CONTACT !!!" if global_is_looking else "Observing...")
                cv2.putText(ann_frame, status_text, (10, 72),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                cv2.putText(ann_frame, f"Current: {current_vlm_action}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

                cv2.imshow("PC External View", ann_frame)
                cv2.imshow("MiRo Right Eye", ann_m_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    usb_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
