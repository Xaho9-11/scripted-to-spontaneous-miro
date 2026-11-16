#!/usr/bin/env python3
"""
MiRo-E animatronics engine (behaviour director): turns behaviour commands into joint, ear, tail and light motion

Changes in this version:
  [1] new A-series actions: a3_neck_roll / a6_slow_stretch_blink / a7_curious_side / a8_wide_scan
  [2] much larger amplitude for happy actions: petting_aftermath pan +/-0.7, b6 pan +/-0.45
  [3] gaze_curious_reaction gets a second phase with a double nod
  [4] minimum sleep protection: wake commands are ignored for 30-60 s after entering c2
"""

import rospy, json, math, threading, time, random
from std_msgs.msg import String, Float32MultiArray, UInt32MultiArray
from sensor_msgs.msg import JointState

RATE = 50
DT   = 1.0 / RATE

ALPHAS = {
    "tilt": 0.04, "lift": 0.04, "pan": 0.05, "pitch": 0.04,
    "eyelid": 0.10, "ear": 0.12, "tail_droop": 0.07, "tail_wag": 0.10,
}
MICRO_DITHER = 0.0003
DITHER_FREQ  = 10.0
# lift:  small = head high (neck retracted), large = head low (neck extended); LIFT_MID=0.594
# pitch: negative = head up, positive = head down
LIFT_MID  = 0.594
LIFT_MIN  = 0.14
LIFT_MAX  = 1.047
PITCH_MID = -0.122

DEFAULT_POSE = {
    "tilt": 0.0, "lift": LIFT_MID, "pan": 0.0, "pitch": 0.03,
    "eyelid": 0.0, "ear": 0.2, "tail_droop": 0.2, "tail_wag": 0.5
}

# sleep timer parameters
SLEEP_ONSET_MIN   = 35.0
SLEEP_ONSET_MAX   = 75.0
C1_DUR_MIN        = 6.0
C1_DUR_MAX        = 9.0
MIN_SLEEP_DUR_MIN = 60.0    # c2 minimum protection: 60 s (was 30 s)
MIN_SLEEP_DUR_MAX = 120.0   # c2 maximum protection: 2 min (was 60 s)
MAX_SLEEP_CYCLES  = 2       # maximum consecutive sleep cycles
FORCE_AWAKE_MIN   = 120.0   # minimum forced-awake time after the limit
FORCE_AWAKE_MAX   = 180.0   # maximum forced-awake time

# actions used when forcing the robot awake (clearly visible)
FORCE_AWAKE_ATOMS = ["a3_neck_roll", "a8_wide_scan", "a7_curious_side",
                     "b2_ambient_scan", "a6_slow_stretch_blink"]

BIOLOGICAL_FATIGUE_DUR = 120.0   # after leaving c2 for longer than this, the robot gets drowsy again (s)

SLEEP_ACTIONS = {"c1_getting_drowsy", "c2_sleep_snore"}
WAKE_ACTIONS  = {
    "b1_curious_tilt", "b2_ambient_scan", "b3_soft_stretch",
    "b4_double_nod", "b5_attention_shift", "b6_happy_wobble",
    "petting_reaction", "petting_aftermath", "gaze_curious_reaction",
    "c3_sudden_wake", "a3_neck_roll", "a6_slow_stretch_blink",
    "a7_curious_side", "a8_wide_scan",
}


def make_color(r, g, b):
    r = max(0, min(255, int(r))); g = max(0, min(255, int(g))); b = max(0, min(255, int(b)))
    return 0xFF000000 | (r << 16) | (g << 8) | b


class Blinker:
    def __init__(self):
        self.next_blink  = time.time() + random.uniform(2.0, 5.0)
        self.is_blinking = False
        self.blink_end   = 0.0

    def update(self, t):
        if not self.is_blinking and t > self.next_blink:
            self.is_blinking = True
            self.blink_end   = t + 0.15
        if self.is_blinking:
            if t > self.blink_end:
                self.is_blinking = False
                self.next_blink  = t + random.uniform(3.0, 7.0)
                return 0.0
            return 1.0
        return 0.0


class SleepScheduler:
    def __init__(self):
        self._last_active       = time.time()
        self._onset_delay       = self._rand_onset()
        self._c1_entered        = None
        self._c1_dur            = self._rand_c1()
        self._c2_entered        = None
        self._min_c2_dur        = 0.0
        self._cycle_count       = 0
        self._force_awake_until = 0.0
        self._last_sleep_end    = time.time()   # start of the biological-fatigue timer

    @staticmethod
    def _rand_onset(): return random.uniform(SLEEP_ONSET_MIN, SLEEP_ONSET_MAX)
    @staticmethod
    def _rand_c1():    return random.uniform(C1_DUR_MIN, C1_DUR_MAX)
    @staticmethod
    def _rand_c2():    return random.uniform(MIN_SLEEP_DUR_MIN, MIN_SLEEP_DUR_MAX)

    def wake(self):
        self._last_active = time.time()
        self._onset_delay = self._rand_onset()
        self._c1_entered  = None
        self._c1_dur      = self._rand_c1()
        self._c2_entered  = None

    def on_wake_from_c2(self):
        self._cycle_count   += 1
        self._last_sleep_end = time.time()   # reset the biological-fatigue timer
        print(f" 😴 sleep cycle {self._cycle_count} done (limit {MAX_SLEEP_CYCLES})")
        if self._cycle_count >= MAX_SLEEP_CYCLES:
            dur = random.uniform(FORCE_AWAKE_MIN, FORCE_AWAKE_MAX)
            self._force_awake_until = time.time() + dur
            self._cycle_count = 0
            forced = random.choice(FORCE_AWAKE_ATOMS)
            print(f" 🌞 sleep limit reached -> forced awake for {dur:.0f}s, first: {forced}")
            return forced
        return None

    def is_sleep_protected(self):
        if self._c2_entered is None: return False
        return time.time() - self._c2_entered < self._min_c2_dur

    def is_force_awake(self):
        return time.time() < self._force_awake_until

    def tick(self, current_action):
        now = time.time()
        if current_action == "c2_sleep_snore":
            if self._c2_entered is None:
                self._c2_entered = now
                self._min_c2_dur = self._rand_c2()
                print(f" 💤 deep-sleep protection {self._min_c2_dur:.0f}s started")
            return None
        if self._c2_entered is not None:
            self._c2_entered = None
        if current_action == "c1_getting_drowsy":
            if self._c1_entered is None: self._c1_entered = now
            if now - self._c1_entered >= self._c1_dur:
                print(f" 💤 c1 {self._c1_dur:.1f}s done -> c2")
                return "c2_sleep_snore"
            return None
        self._c1_entered = None
        if self.is_force_awake():
            return None
        # biological fatigue: drowsy again after leaving c2 for longer than the threshold
        awake_dur = now - self._last_sleep_end
        if awake_dur >= BIOLOGICAL_FATIGUE_DUR:
            self._last_sleep_end = now
            print(f" 😪 awake {awake_dur:.0f}s -> biological fatigue triggers c1")
            return "c1_getting_drowsy"
        # ordinary idle timeout
        elapsed = now - self._last_active
        if elapsed >= self._onset_delay:
            print(f" 💤 idle {elapsed:.0f}s (threshold {self._onset_delay:.0f}s) -> c1")
            return "c1_getting_drowsy"
        return None

    @staticmethod
    def _rand_onset(): return random.uniform(SLEEP_ONSET_MIN, SLEEP_ONSET_MAX)
    @staticmethod
    def _rand_c1():    return random.uniform(C1_DUR_MIN, C1_DUR_MAX)
    @staticmethod
    def _rand_c2():    return random.uniform(MIN_SLEEP_DUR_MIN, MIN_SLEEP_DUR_MAX)

    def wake(self):
        self._last_active = time.time()
        self._onset_delay = self._rand_onset()
        self._c1_entered  = None
        self._c1_dur      = self._rand_c1()
        self._c2_entered  = None

    def on_wake_from_c2(self):
        """Called on waking from c2: counts the cycle; beyond the limit the robot is forced awake."""
        self._cycle_count += 1
        print(f" 😴 sleep cycle {self._cycle_count} done (limit {MAX_SLEEP_CYCLES})")
        if self._cycle_count >= MAX_SLEEP_CYCLES:
            dur = random.uniform(FORCE_AWAKE_MIN, FORCE_AWAKE_MAX)
            self._force_awake_until = time.time() + dur
            self._cycle_count = 0
            forced = random.choice(FORCE_AWAKE_ATOMS)
            print(f" 🌞 sleep limit reached -> forced awake for {dur:.0f}s, first: {forced}")
            return forced   # the action to run now
        return None

    def is_sleep_protected(self):
        if self._c2_entered is None: return False
        return time.time() - self._c2_entered < self._min_c2_dur

    def is_force_awake(self):
        return time.time() < self._force_awake_until

    def tick(self, current_action):
        now = time.time()
        if current_action == "c2_sleep_snore":
            if self._c2_entered is None:
                self._c2_entered = now
                self._min_c2_dur = self._rand_c2()
                print(f" 💤 deep-sleep protection {self._min_c2_dur:.0f}s started")
            return None
        if self._c2_entered is not None:
            self._c2_entered = None   # clear the record after leaving c2
        if current_action == "c1_getting_drowsy":
            if self._c1_entered is None: self._c1_entered = now
            if now - self._c1_entered >= self._c1_dur:
                print(f" 💤 c1 {self._c1_dur:.1f}s done -> c2")
                return "c2_sleep_snore"
            return None
        self._c1_entered = None
        # no sleep while forced awake
        if self.is_force_awake():
            rem = self._force_awake_until - now
            if int(rem) % 30 == 0 and rem % 1.0 < 0.05:
                print(f" 🌞 forced awake, {rem:.0f}s left")
            return None
        elapsed = now - self._last_active
        if elapsed >= self._onset_delay:
            print(f" 💤 idle {elapsed:.0f}s (threshold {self._onset_delay:.0f}s) -> c1")
            return "c1_getting_drowsy"
        return None


class MiroAnimatronics:
    def __init__(self):
        rospy.init_node('miro_animatronics_engine', anonymous=True)
        base = "/miro"
        self.pub_kin = rospy.Publisher(f"{base}/control/kinematic_joints", JointState, queue_size=1)
        self.pub_cos = rospy.Publisher(f"{base}/control/cosmetic_joints",  Float32MultiArray, queue_size=1)
        self.pub_led = rospy.Publisher(f"{base}/control/illum",            UInt32MultiArray, queue_size=1)

        self.current_action    = "a4_focus_mirror"
        self.intensity         = 0.5
        self.led_mode          = "normal"
        self.action_start_time = time.time()
        self.blinker           = Blinker()
        self.sleep_sched       = SleepScheduler()
        self.x                 = DEFAULT_POSE.copy()

        # noise_mode: 'organic'    (Spontaneous condition: multi-frequency life-like background motion)
        #             'mechanical' (Scripted condition: no background motion between actions)
        self.noise_mode = rospy.get_param('~noise_mode', 'organic')

        rospy.Subscriber(f"{base}/behavior_cmd", String, self.cmd_callback)
        threading.Thread(target=self.loop, daemon=True).start()
        print(" 🤖 MiRo director: differentiated actions + fade-out + biological-fatigue sleep")
        print(f"    noise mode: {self.noise_mode.upper()}"
              f"  ({'organic, multi-frequency' if self.noise_mode == 'organic' else 'mechanical'})")
        print(f"    biological fatigue: drowsy again after {BIOLOGICAL_FATIGUE_DUR:.0f}s")

    def _set_action(self, name):
        if name != self.current_action:
            self.current_action    = name
            self.action_start_time = time.time()

    def cmd_callback(self, msg):
        try:
            data       = json.loads(msg.data)
            new_action = data.get("action", "idle")
            if new_action == "idle": return

            if new_action in WAKE_ACTIONS:
                # ignore wake commands during deep-sleep protection
                if self.sleep_sched.is_sleep_protected():
                    rem = (self.sleep_sched._min_c2_dur
                           - (time.time() - self.sleep_sched._c2_entered))
                    print(f" 🔒 deep-sleep protection ({rem:.0f}s left), ignoring: {new_action}")
                    return
                # waking from c2: count the cycle; beyond the limit, run the forced wake action
                if self.current_action == "c2_sleep_snore":
                    forced = self.sleep_sched.on_wake_from_c2()
                    self.sleep_sched.wake()
                    if forced:
                        self._set_action(forced)
                        self.intensity = 0.8
                        return
                elif self.current_action in SLEEP_ACTIONS:
                    print(f" ⚡ wake: {self.current_action} -> {new_action}")
                self.sleep_sched.wake()

            # while forced awake, block all sleep commands (c3_sudden_wake is still allowed)
            if (self.sleep_sched.is_force_awake()
                    and new_action in SLEEP_ACTIONS):
                print(f" 🌞 forced awake, ignoring sleep command: {new_action}")
                return

            self._set_action(new_action)
            self.led_mode  = data.get("led_mode", "normal")
            self.intensity = max(0.0, min(1.0, float(data.get("intensity", 0.5))))
        except Exception:
            pass

    def calculate_targets(self, t_now):
        t_act = t_now - self.action_start_time
        tgt   = DEFAULT_POSE.copy()
        bb    = self.blinker.update(t_now)
        tgt["eyelid"] = bb
        I = self.intensity   # shorthand

        # -- physical reflexes ---------------------------------------------------
        if self.current_action == "gaze_curious_reaction":
            # Phase 1 (0-1.5 s): notices eye contact, tilts the head
            # Phase 2 (1.5 s+):  two clear nods
            if t_act < 1.5:
                p = min(1.0, t_act / 0.6)
                tgt["lift"]  = 0.42 + 0.05 * math.sin(t_act * 4.0)
                tgt["tilt"]  = 0.32 * p * I
                tgt["pan"]   = 0.18 * p * I
                tgt["pitch"] = -0.05
                tgt["ear"]   = 0.88 * I
                tgt["tail_droop"] = 0.60
                tgt["tail_wag"]   = 0.5 + 0.28 * math.sin(t_act * 10.0) * I
            else:
                nt = t_act - 1.5
                nod = abs(math.sin(nt * math.pi * 2.8)) if nt < 0.75 else 0.0
                tgt["lift"]  = LIFT_MID + 0.12 * nod * I
                tgt["pitch"] = 0.03 + 0.20 * nod * I
                tgt["pan"]   = 0.06 * math.sin(nt * 3.0) * I
                tgt["tilt"]  = 0.08 * I
                tgt["ear"]   = 0.75 * I
                tgt["tail_droop"] = 0.58
                tgt["tail_wag"]   = 0.5 + 0.22 * math.sin(t_act * 8.0) * I

        elif self.current_action == "petting_reaction":
            tgt["lift"]  = LIFT_MID + 0.05
            tgt["pitch"] = 0.04
            tgt["ear"]   = 0.3 + 0.5 * math.sin(t_act * 10.0) * I
            tgt["tail_droop"] = 0.94
            tgt["tail_wag"]   = 0.5 + 0.5 * math.sin(t_act * 30.0) * I
            tgt["eyelid"] = max(bb, 0.55 * (1.0 + math.sin(t_act * 3.0)) * 0.5)

        elif self.current_action == "petting_aftermath":
            # [Fix 2] wide side-to-side head shake: pan +/-0.70, tilt +/-0.22
            decay = max(0.0, 1.0 - t_act / 3.5)
            f     = t_act * 11.0
            tgt["pan"]   = 0.70 * math.sin(f) * min(1.0, t_act * 2.5) * decay * I
            tgt["tilt"]  = 0.22 * math.cos(f * 0.8) * decay * I
            tgt["lift"]  = LIFT_MID + 0.04 - 0.08 * (1.0 - decay)
            tgt["pitch"] = 0.02 * (1.0 - decay)
            tgt["ear"]   = 0.35 + 0.50 * decay
            tgt["tail_droop"] = 0.90 - 0.38 * (1.0 - decay)
            tgt["tail_wag"]   = 0.5 + 0.50 * decay * math.sin(t_act * (28*decay + 4))
            tgt["eyelid"] = max(bb, 0.20 * decay)

        # -- C series: sleep chain ------------------------------------------------
        elif self.current_action == "c1_getting_drowsy":
            p   = min(1.0, t_act / 5.0)
            nod = 0.06 * math.sin(t_act * 7.0) if 2.0 < t_act < 4.0 else 0.0
            tgt["lift"]  = LIFT_MID + 0.35 * p + nod
            tgt["pitch"] = 0.02 + 0.11 * p
            tgt["eyelid"]= max(bb, p)
            tgt["ear"]   = 0.2 - 0.2 * p
            tgt["tail_droop"] = 0.12

        elif self.current_action == "c2_sleep_snore":
            br = 0.012 * math.sin(t_act * 1.5)
            tgt["lift"]  = 0.95 + br
            tgt["pitch"] = 0.13
            tgt["eyelid"]= 1.0
            tgt["tail_wag"] = 0.5 + 0.06 * math.sin(t_act * 0.42)
            # ears twitch now and then: two frequencies multiplied give random-looking pulses
            ear_flutter = abs(math.sin(t_act * 0.29) * math.sin(t_act * 1.83))
            tgt["ear"]       = 0.02 + 0.38 * ear_flutter
            # tail moves slightly now and then
            tail_flutter = abs(math.sin(t_act * 0.17) * math.sin(t_act * 0.71))
            tgt["tail_droop"] = 0.04 + 0.16 * tail_flutter

        elif self.current_action == "c3_sudden_wake":
            tgt["lift"]  = 0.28
            tgt["pitch"] = -0.10
            tgt["eyelid"]= 0.0
            tgt["ear"]   = 1.0 * I
            if t_act < 1.8:
                tgt["pan"]  = 0.60 * math.sin(t_act * 14.0) * I
                tgt["tilt"] = 0.12 * math.cos(t_act * 7.0) * I

        # -- A series: clearly visible actions (chosen by the VLM) -----------------
        elif self.current_action == "a3_neck_roll":
            # neck circle: pan leads tilt by 90 degrees, slightly faster, fades out over 6-8 s
            ang  = t_act * 0.95          # a little faster than 0.78
            fade = min(1.0, t_act / 1.0) * max(0.0, (8.0 - t_act) / 2.0)
            fade = max(0.0, min(1.0, fade))
            tgt["pan"]   = 0.62 * math.sin(ang) * I * fade
            tgt["tilt"]  = 0.30 * math.cos(ang) * I * fade   # 90-degree phase shift = circle
            tgt["lift"]  = LIFT_MID + 0.08 * math.cos(ang)
            tgt["pitch"] = 0.03
            tgt["eyelid"]= max(bb, 0.08)
            tgt["ear"]   = 0.40 + 0.18 * math.sin(ang)
            tgt["tail_droop"] = 0.35
            # no background motion during neck_roll (it would distort the circle)
            return tgt   # return early, skipping the background layer

        elif self.current_action == "a6_slow_stretch_blink":
            # stretch: neck extends forward (lift up), eyes close, then retracts
            if t_act < 2.5:
                p = t_act / 2.5
                tgt["lift"]   = LIFT_MID + 0.32 * p          # neck forward = head low
                tgt["pitch"]  = 0.03 + 0.07 * p              # head slightly down
                tgt["eyelid"] = p * 0.95
                tgt["ear"]    = 0.22 + 0.28 * p
            elif t_act < 3.2:
                tgt["lift"]   = LIFT_MID + 0.32
                tgt["pitch"]  = 0.10
                tgt["eyelid"] = 0.95
                tgt["ear"]    = 0.50
            else:
                p = min(1.0, (t_act - 3.2) / 0.5)
                tgt["lift"]   = LIFT_MID + 0.32 * (1.0 - p)
                tgt["pitch"]  = 0.10 - 0.07 * p
                tgt["eyelid"] = max(bb, 0.95 * (1.0 - p))
                tgt["ear"]    = 0.50 - 0.28 * p
            tgt["tail_droop"] = 0.30

        elif self.current_action == "a7_curious_side":
            # tilt: quick lean -> hold -> slow return (tilt-dominated)
            if t_act < 0.7:
                p = t_act / 0.7
                tgt["tilt"] = 0.42 * p * I
                tgt["pan"]  = 0.10 * p * I    # very little pan
            elif t_act < 3.5:
                tgt["tilt"] = 0.42 * I
                tgt["pan"]  = 0.10 * I
            else:
                p = min(1.0, (t_act - 3.5) / 0.9)
                tgt["tilt"] = 0.42 * (1.0 - p) * I
                tgt["pan"]  = 0.10 * (1.0 - p) * I
            tgt["lift"]  = LIFT_MID + 0.05
            tgt["pitch"] = 0.02
            tgt["eyelid"]= max(bb, 0.05)
            tgt["ear"]   = 0.65 * I
            tgt["tail_droop"] = 0.55

        elif self.current_action == "a8_wide_scan":
            # single 180-degree scan with smoothstep easing, one side to the other in 8 s
            # unlike b2: one pass, slow and attentive, like checking every corner
            SCAN_DUR = 8.0
            p    = min(1.0, t_act / SCAN_DUR)
            sp   = p * p * (3.0 - 2.0 * p)           # smoothstep
            pan  = 0.90 * (2.0 * sp - 1.0) * I       # -0.90 → +0.90
            tgt["pan"]   = pan
            tgt["lift"]  = LIFT_MID + 0.05 * math.sin(math.pi * p)  # slight lift midway
            tgt["tilt"]  = 0.04 * math.sin(math.pi * p)             # slight tilt (natural scan)
            tgt["pitch"] = 0.03
            tgt["eyelid"]= max(bb, 0.05)
            tgt["ear"]   = 0.68
            tgt["tail_droop"] = 0.45

        # -- B series: active interaction (larger amplitudes overall) -----------
        elif self.current_action == "b1_curious_tilt":
            tgt["lift"]  = 0.42
            tgt["tilt"]  = 0.42 * I
            tgt["pan"]   = 0.25 * math.sin(t_act * 3.0) * I
            tgt["pitch"] = -0.05
            tgt["eyelid"]= 0.0
            tgt["ear"]   = 0.92 * I
            tgt["tail_droop"] = 0.82
            tgt["tail_wag"]   = 0.5 + 0.28 * math.sin(t_act * 8.0) * I

        elif self.current_action == "b2_ambient_scan":
            # fast back-and-forth scan (unlike a8: oscillating, quick, tense)
            # amplitude fades after 6 s, so it does not loop forever
            fade = min(1.0, max(0.0, (7.0 - t_act) / 2.0))
            sc   = math.sin(t_act * 1.3)
            tgt["lift"]  = LIFT_MID + 0.10 * abs(sc)   # lift changes slightly while turning
            tgt["pan"]   = 0.72 * sc * I * fade
            tgt["tilt"]  = 0.10 * math.sin(t_act * 2.6) * fade   # tilt follows at twice the frequency
            tgt["pitch"] = 0.03 + 0.03 * abs(sc)
            tgt["eyelid"]= max(bb, 0.08)
            tgt["ear"]   = 0.82
            tgt["tail_droop"] = 0.58

        elif self.current_action == "b3_soft_stretch":
            st = math.sin(t_act * 1.2) if t_act < 2.6 else 0.0
            tgt["lift"]  = LIFT_MID + 0.42 * max(0.0, st) * I
            tgt["pan"]   = 0.14 * math.sin(t_act * 2.5) * I
            tgt["pitch"] = 0.03 + 0.10 * max(0.0, st) * I
            tgt["eyelid"]= max(bb, 0.80 * max(0.0, st))
            tgt["ear"]   = 0.25 + 0.42 * max(0.0, st)
            tgt["tail_droop"] = 0.25 + 0.55 * max(0.0, st)

        elif self.current_action == "b4_double_nod":
            nod = abs(math.sin(t_act * math.pi * 2.5)) if t_act < 0.9 else 0.0
            tgt["lift"]  = LIFT_MID + 0.12 * nod * I
            tgt["pitch"] = 0.03 + 0.22 * nod * I
            tgt["pan"]   = 0.10 * math.sin(t_act * 5.0) * nod * I
            tgt["eyelid"]= max(bb, 0.18)
            tgt["ear"]   = 0.68
            tgt["tail_droop"] = 0.55

        elif self.current_action == "b5_attention_shift":
            sh = min(1.0, t_act * 2.5)
            tgt["lift"]  = LIFT_MID - 0.12 * sh
            tgt["pan"]   = -0.70 * sh * I
            tgt["tilt"]  = 0.18 * sh * I
            tgt["pitch"] = -0.03 * sh
            tgt["eyelid"]= max(bb, 0.06)
            tgt["ear"]   = 0.88 * sh
            tgt["tail_droop"] = 0.68

        elif self.current_action == "b6_happy_wobble":
            # happy wobble: tilt-dominated (sideways sway), pan secondary at another frequency
            # fades after 5 s, so it does not wobble forever
            fade = min(1.0, max(0.0, (6.0 - t_act) / 2.0))
            wb   = t_act * 10.0
            tgt["tilt"]  = 0.42 * math.cos(wb) * I * fade          # tilt dominates
            tgt["pan"]   = 0.20 * math.sin(wb * 1.4) * I * fade    # pan, asynchronous support
            tgt["lift"]  = LIFT_MID + 0.07 * math.sin(wb * 0.5) * fade
            tgt["pitch"] = 0.03 - 0.04 * math.cos(wb * 0.5) * fade
            tgt["ear"]   = min(1.0, 0.92 * I + 0.08 * math.sin(wb * 2) * fade)
            tgt["tail_droop"] = 0.96
            tgt["tail_wag"]   = 0.5 + 0.52 * math.sin(t_act * 24.0) * I * fade

        # -- background life motion (muted during sleep and aftermath) -----------
        if self.current_action not in {"c1_getting_drowsy", "c2_sleep_snore",
                                        "petting_aftermath"}:
            t = t_now
            if self.noise_mode == 'mechanical':
                # mechanical mode freezes the background layer: idle motors hold still, a deliberately stiff feel
                pass 
            else:
                # multi-frequency sum: organic, life-like (default)
                tgt["lift"]     += 0.025 * math.sin(t * 0.28) + 0.010 * math.sin(t * 0.67)
                tgt["pan"]      += 0.035 * math.sin(t * 0.52) + 0.015 * math.sin(t * 1.10)
                tgt["tilt"]     += 0.018 * math.sin(t * 0.41 + 1.2)
                tgt["pitch"]    += 0.008 * math.sin(t * 0.44)
                tgt["ear"]      += 0.055 * math.sin(t * 0.31) + 0.025 * math.sin(t * 0.83)
                tgt["tail_wag"] += 0.040 * math.cos(t * 0.35) + 0.015 * math.cos(t * 0.72)

        return tgt

    def update_leds(self, t):
        colors = []
        if self.led_mode == "happy":
            for i in range(6):
                hue = (t * 0.5 + i / 6.0) % 1.0
                h = hue * 6; x = 1 - abs(h % 2 - 1)
                if   h < 1: r,g,b = 1,x,0
                elif h < 2: r,g,b = x,1,0
                elif h < 3: r,g,b = 0,1,x
                elif h < 4: r,g,b = 0,x,1
                elif h < 5: r,g,b = x,0,1
                else:        r,g,b = 1,0,x
                colors.append(make_color(r*255, g*255, b*255))
        elif self.led_mode == "petting":
            beat = (math.exp(-((t%1.0-0.10)**2)/0.005) +
                    math.exp(-((t%1.0-0.35)**2)/0.005))
            colors = [make_color(int(min(1.0,beat)*255), 0, 0)] * 6
        elif self.led_mode == "focused":
            v = (math.sin(t * 1.5) + 1) / 2
            colors = [make_color(0, int(v*80), int(v*255))] * 6
        else:
            for i in range(6):
                ph = (math.sin(t * 0.8 + i * 0.5) + 1) / 2
                colors.append(make_color(int(ph*150), 20, int((1-ph)*150)))
        msg = UInt32MultiArray()
        msg.data = colors
        self.pub_led.publish(msg)

    def loop(self):
        rate     = rospy.Rate(RATE)
        t_dither = 0.0
        while not rospy.is_shutdown():
            forced = self.sleep_sched.tick(self.current_action)
            if forced: self._set_action(forced)

            t_now = time.time()
            tgt   = self.calculate_targets(t_now)

            for k in self.x:
                a = ALPHAS.get(k, 0.06)
                self.x[k] = self.x[k] * (1-a) + tgt[k] * a

            d = MICRO_DITHER * math.sin(2 * math.pi * DITHER_FREQ * t_dither)

            kin = JointState()
            kin.header.stamp = rospy.Time.now()
            kin.name         = ["tilt", "lift", "yaw", "pitch"]
            kin.position     = [self.x["tilt"]+d, self.x["lift"]+d,
                                 self.x["pan"]+d,  self.x["pitch"]]
            self.pub_kin.publish(kin)

            cos = Float32MultiArray()
            ear_l = (0.94*self.intensity if self.current_action in
                     {"b1_curious_tilt","gaze_curious_reaction"} else self.x["ear"])
            ear_r = (0.06*self.intensity if self.current_action in
                     {"b1_curious_tilt","gaze_curious_reaction"} else self.x["ear"])
            cos.data = [self.x["tail_droop"], self.x["tail_wag"],
                        self.x["eyelid"], self.x["eyelid"], ear_l, ear_r]
            self.pub_cos.publish(cos)

            self.update_leds(t_now)
            t_dither += DT
            rate.sleep()


if __name__ == '__main__':
    try:
        MiroAnimatronics()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
