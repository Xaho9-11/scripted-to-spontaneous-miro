#!/usr/bin/env python3
"""
MiRo Experiment Control Center — Premium Unified Dark UI (v2.0)
"""

import sys
import os
import json
import subprocess
import signal
import time
import threading
import traceback
from collections import deque

try:
    import rospy
    from std_msgs.msg import String
    from sensor_msgs.msg import CompressedImage
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
    print("[Warn] rospy not available — UI will start in offline mode")

import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets

HERE         = os.path.dirname(os.path.abspath(__file__))
PROFILE_PATH = os.path.join(HERE, "tuner_profile.json")
SPONT_SCRIPT = os.path.join(HERE, "spontaneous_pipeline.py")
MECH_SCRIPT  = os.path.join(HERE, "scripted_pipeline.py")


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline subprocess manager
# ═══════════════════════════════════════════════════════════════════════════════
class PipelineManager:
    def __init__(self):
        self.proc = None
        self.mode = None

    def is_running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, mode):
        if self.is_running():
            self.stop()
        script = SPONT_SCRIPT if mode == "spontaneous" else MECH_SCRIPT
        if not os.path.exists(script):
            return False, f"Script not found: {script}"
        try:
            self.proc = subprocess.Popen(
                ["python3", script],
                stdout=None, stderr=None,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )
            self.mode = mode
            return True, f"Started {mode} (pid {self.proc.pid})"
        except Exception as e:
            return False, str(e)

    def stop(self):
        if not self.is_running():
            self.mode = None
            return
        try:
            if hasattr(os, 'killpg'):
                os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
            else:
                self.proc.send_signal(signal.SIGINT)
            self.proc.wait(timeout=4.0)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        finally:
            self.proc = None
            self.mode = None


# ═══════════════════════════════════════════════════════════════════════════════
# Tuner Profile Helpers
# ═══════════════════════════════════════════════════════════════════════════════
PROFILE_PRESETS = {
    "Watchful Companion": {
        "calm_preference":   "Stay quietly attentive. Subtle, slow presence. This user finds comfort in knowing you are there.",
        "active_preference": "Stay steady and still. Low intensity almost always right. Do not add to their load.",
    },
    "Pattern Breaker": {
        "calm_preference":   "Be noticeable but not constant. Occasional clear gestures are welcome.",
        "active_preference": "More active — a clear, deliberate gesture can help break a thought loop. Higher intensity acceptable.",
    },
    "Ambient Presence": {
        "calm_preference":   "Barely move. Extremely quiet. Any movement that draws attention feels like intrusion.",
        "active_preference": "Go completely still or near-still. Any animation is unwanted. Idle is almost always correct.",
    },
}

def guess_profile_type(calm_lma, active_lma):
    ca = active_lma
    if all(v == 0 for v in ca.values()):
        return "Ambient Presence"
    if ca.get("space", 0) >= 70 and ca.get("weight", 0) >= 30:
        return "Pattern Breaker"
    if ca.get("space", 0) >= 50 and ca.get("weight", 0) < 30:
        return "Watchful Companion"
    avg = sum(ca.values()) / max(len(ca), 1)
    return "Ambient Presence" if avg < 20 else "Watchful Companion"

def parse_tuner_text(text):
    text = text.strip()
    if not text: return []
    try:
        d = json.loads(text)
        return d if isinstance(d, list) else [d]
    except json.JSONDecodeError:
        pass
    records, depth, buf = [], 0, []
    for ch in text:
        if ch == "{": depth += 1
        if depth > 0: buf.append(ch)
        if ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                try: records.append(json.loads("".join(buf)))
                except json.JSONDecodeError: pass
                buf = []
    return records

def lma_map(raw):
    return {
        "weight": raw.get("weight_energy", 0),
        "time":   raw.get("time_tempo",    0),
        "flow":   raw.get("flow_spontaneity", 0),
        "space":  raw.get("space_focus",   0),
    }

def write_profile_from_tuner(text, participant_id=None, override_type=None):
    records = parse_tuner_text(text)
    if not records: return False, "Could not parse any JSON records."
    if not participant_id:
        ids = [r.get("participantId") for r in records if r.get("participantId")]
        if not ids: return False, "No participantId found in JSON."
        participant_id = max(set(ids), key=ids.count)
    calm = next((r for r in records if r.get("participantId") == participant_id and r.get("contextMode") == "calm"), None)
    active = next((r for r in records if r.get("participantId") == participant_id and r.get("contextMode") == "active"), None)
    if not calm or not active: return False, f"Need both 'calm' and 'active' records for {participant_id}."
    calm_lma   = lma_map(calm["lmaParameters"])
    active_lma = lma_map(active["lmaParameters"])
    ptype      = override_type or guess_profile_type(calm_lma, active_lma)
    prefs      = PROFILE_PRESETS[ptype]
    profile = {
        "participantId":  participant_id,
        "profileType":    ptype,
        "mentalMetaphor": (calm.get("mentalMetaphor","") + " / " + active.get("mentalMetaphor","")).strip(" /"),
        "calm_pref":         prefs["calm_preference"],
        "active_pref":       prefs["active_preference"],
        "calm_preference":   prefs["calm_preference"],
        "active_preference": prefs["active_preference"],
        "lma_calm":   calm_lma,
        "lma_active": active_lma,
    }
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    return True, profile


# ═══════════════════════════════════════════════════════════════════════════════
# ROS bridge
# ═══════════════════════════════════════════════════════════════════════════════
class RosBridge(QtCore.QObject):
    ext_frame_received = QtCore.pyqtSignal(np.ndarray)
    eye_frame_received = QtCore.pyqtSignal(np.ndarray)
    ui_data_received   = QtCore.pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        if not HAS_ROS: return
        rospy.init_node("miro_ui_qt", anonymous=True, disable_signals=True)
        rospy.Subscriber("/miro/ui_frame_ext", CompressedImage, self._cb_ext, queue_size=1)
        rospy.Subscriber("/miro/ui_frame_eye", CompressedImage, self._cb_eye, queue_size=1)
        rospy.Subscriber("/miro/ui_data",      String,          self._cb_ui,  queue_size=1)

    def _cb_ext(self, msg):
        try:
            arr = np.frombuffer(msg.data, np.uint8)
            f = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if f is not None: self.ext_frame_received.emit(f)
        except Exception: pass

    def _cb_eye(self, msg):
        try:
            arr = np.frombuffer(msg.data, np.uint8)
            f = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if f is not None: self.eye_frame_received.emit(f)
        except Exception: pass

    def _cb_ui(self, msg):
        try: self.ui_data_received.emit(json.loads(msg.data))
        except Exception: pass


# ═══════════════════════════════════════════════════════════════════════════════
# Premium Dark UI Stylesheet
# ═══════════════════════════════════════════════════════════════════════════════
STYLESHEET = """
QMainWindow, QWidget {
    background-color: #121214;
    color: #e1e1e6;
    font-family: -apple-system, "Segoe UI", "SF Pro Display", sans-serif;
    font-size: 13px;
}

QFrame#card {
    background-color: #1e1e22;
    border-radius: 12px;
    border: 1px solid #2d2d30;
}

QLabel#title {
    font-size: 22px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.5px;
}

QLabel#subtitle {
    font-size: 11px;
    color: #8e8e93;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 700;
}

QLabel#kpi_value {
    font-size: 18px;
    font-weight: 700;
    color: #0a84ff;
    font-family: "SF Mono", Menlo, monospace;
}

QLabel#kpi_label {
    font-size: 10px;
    color: #767680;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
}

QLabel#section_label {
    font-size: 13px;
    color: #f2f2f7;
    font-weight: 600;
}

QPushButton {
    background-color: #2c2c2e;
    color: #e5e5ea;
    border: 1px solid #3a3a3c;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 600;
}
QPushButton:hover   { background-color: #3a3a3c; border-color: #48484a; }
QPushButton:pressed { background-color: #1c1c1e; }
QPushButton:disabled { color: #48484a; border-color: #2c2c2e; }

QPushButton#primary {
    background-color: #0a84ff;
    color: #ffffff;
    border: none;
}
QPushButton#primary:hover { background-color: #2993ff; }
QPushButton#primary:pressed { background-color: #0066cc; }

QPushButton#danger {
    background-color: #1e1e22;
    color: #ff453a;
    border: 1px solid #ff453a;
}
QPushButton#danger:hover { background-color: #ff453a; color: #ffffff; }

QPushButton#mode_btn {
    padding: 11px 20px;
    font-size: 13px;
}
QPushButton#mode_btn[active="true"] {
    background-color: #0a84ff;
    color: #ffffff;
    border: none;
}

QPlainTextEdit, QLineEdit {
    background-color: #121214;
    border: 1px solid #2d2d30;
    border-radius: 8px;
    padding: 8px 10px;
    color: #e1e1e6;
    font-family: "SF Mono", "Menlo", monospace;
    font-size: 12px;
}
QPlainTextEdit:focus, QLineEdit:focus { border: 1px solid #0a84ff; }

QComboBox {
    background-color: #2c2c2e;
    border: 1px solid #3a3a3c;
    border-radius: 8px;
    padding: 6px 12px;
    color: #e1e1e6;
}

QProgressBar {
    border: none;
    background-color: #2c2c2e;
    border-radius: 4px;
    text-align: right;
    height: 8px;
}
QProgressBar::chunk {
    background-color: #64d2ff;
    border-radius: 4px;
}
QProgressBar[active="true"]::chunk {
    background-color: #30d158;
}

QLabel#status_pill {
    background-color: #2c2c2e;
    border-radius: 12px;
    padding: 5px 14px;
    font-size: 11px;
    font-weight: 700;
    border: 1px solid #3a3a3c;
}
QLabel#status_pill[state="running_spont"] { background-color: #0a84ff; color: white; border: none; }
QLabel#status_pill[state="running_mech"] { background-color: #30d158; color: white; border: none; }
QLabel#status_pill[state="stopped"] { background-color: #2c2c2e; color: #8e8e93; }

QLabel#inspector_badge {
    background-color: #ff9f0a; color: black; border-radius: 4px; padding: 2px 8px; font-size: 10px; font-weight: 700;
}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Camera view widget (Locked bounds to prevent deformation & shaking)
# ═══════════════════════════════════════════════════════════════════════════════
class CameraView(QtWidgets.QLabel):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        # fix: lock the physical size so changing text cannot stretch the layout sideways
        self.setFixedSize(480, 360)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("background-color: #000000; border-radius: 10px; border: 1px solid #2d2d30; color: #545456;")
        self.setText(f"{title}\nwaiting for feed...")
        self._latest = None

    def update_frame(self, frame_bgr):
        # fix: normalise crop and scale before rendering (removed the jitter seen in mechanical mode)
        try:
            frame_resized = cv2.resize(frame_bgr, (480, 360))
            self._latest = frame_resized
            h, w, _ = frame_resized.shape
            rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = QtGui.QImage(rgb.data, w, h, w * 3, QtGui.QImage.Format_RGB888)
            self.setPixmap(QtGui.QPixmap.fromImage(img))
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Main window
# ═══════════════════════════════════════════════════════════════════════════════
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, ros_bridge):
        super().__init__()
        self.setWindowTitle("MiRo Experiment Control Center")
        self.resize(1500, 920)
        self.setStyleSheet(STYLESHEET)

        self.pm        = PipelineManager()
        self.inspector = True
        self.ros       = ros_bridge

        self._build_ui()
        self._wire_ros()
        self._refresh_profile_display()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._heartbeat)
        self.timer.start(500)

    # ───────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(18)

        # ─── LEFT COLUMN ───────────────────────────────────────────────────────
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(14)

        # Header
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("MiRo Control Center")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()
        self.inspector_badge = QtWidgets.QLabel("INSPECTOR")
        self.inspector_badge.setObjectName("inspector_badge")
        header.addWidget(self.inspector_badge)
        self.status_pill = QtWidgets.QLabel("STOPPED")
        self.status_pill.setObjectName("status_pill")
        self.status_pill.setProperty("state", "stopped")
        header.addWidget(self.status_pill)
        left.addLayout(header)

        # Live view card
        cam_card = QtWidgets.QFrame()
        cam_card.setObjectName("card")
        cam_layout = QtWidgets.QVBoxLayout(cam_card)
        cam_layout.setContentsMargins(16, 14, 16, 16)
        cam_layout.setSpacing(10)

        cam_title = QtWidgets.QLabel("LIVE VIDEO STREAMS")
        cam_title.setObjectName("subtitle")
        cam_layout.addWidget(cam_title)

        cams_row = QtWidgets.QHBoxLayout()
        cams_row.setSpacing(14)
        self.cam_ext = CameraView("EXTERNAL WEB_CAM")
        self.cam_eye = CameraView("MIRO ROBOT VISION")
        cams_row.addWidget(self.cam_ext)
        cams_row.addWidget(self.cam_eye)
        cam_layout.addLayout(cams_row)

        labels_row = QtWidgets.QHBoxLayout()
        for txt in ("PARTICIPANT FOCUS VIEW (EXTERNAL)", "ROBOTIC RIGHT EYE FEED"):
            l = QtWidgets.QLabel(txt)
            l.setObjectName("kpi_label")
            l.setAlignment(QtCore.Qt.AlignCenter)
            labels_row.addWidget(l)
        cam_layout.addLayout(labels_row)
        left.addWidget(cam_card)

        # VLM Reasoning card
        vlm_card = QtWidgets.QFrame()
        vlm_card.setObjectName("card")
        vlm_l = QtWidgets.QVBoxLayout(vlm_card)
        vlm_l.setContentsMargins(16, 14, 16, 16)
        vlm_l.setSpacing(8)
        vt = QtWidgets.QLabel("VLM COGNITIVE ENGINE REASONING LOG")
        vt.setObjectName("subtitle")
        vlm_l.addWidget(vt)

        self.lbl_visual  = self._make_reasoning_row("VISUAL COGNITION")
        self.lbl_thought = self._make_reasoning_row("INTERNAL STATE & THOUGHT")
        self.lbl_reason  = self._make_reasoning_row("BEHAVIORAL DECISION REASON")
        vlm_l.addLayout(self.lbl_visual)
        vlm_l.addLayout(self.lbl_thought)
        vlm_l.addLayout(self.lbl_reason)
        left.addWidget(vlm_card)

        root.addLayout(left, stretch=7)

        # ─── RIGHT COLUMN ──────────────────────────────────────────────────────
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(14)

        # Controls
        mode_card = QtWidgets.QFrame()
        mode_card.setObjectName("card")
        mode_l = QtWidgets.QVBoxLayout(mode_card)
        mode_l.setContentsMargins(16, 14, 16, 16)
        mode_l.setSpacing(10)
        mt = QtWidgets.QLabel("PIPELINE EXECUTION")
        mt.setObjectName("subtitle")
        mode_l.addWidget(mt)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(10)
        self.btn_spont = QtWidgets.QPushButton("Spontaneous")
        self.btn_mech  = QtWidgets.QPushButton("Mechanical")
        self.btn_spont.setObjectName("mode_btn")
        self.btn_mech.setObjectName("mode_btn")
        self.btn_spont.clicked.connect(lambda: self._switch_mode("spontaneous"))
        self.btn_mech.clicked.connect(lambda: self._switch_mode("mechanical"))
        btn_row.addWidget(self.btn_spont)
        btn_row.addWidget(self.btn_mech)
        mode_l.addLayout(btn_row)

        self.btn_stop = QtWidgets.QPushButton("Emergency Stop")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.clicked.connect(self._stop_mode)
        mode_l.addWidget(self.btn_stop)
        right.addWidget(mode_card)

        # Participant profile Card
        p_card = QtWidgets.QFrame()
        p_card.setObjectName("card")
        pl = QtWidgets.QVBoxLayout(p_card)
        pl.setContentsMargins(16, 14, 16, 16)
        pl.setSpacing(10)
        pt = QtWidgets.QLabel("PARTICIPANT PROFILE")
        pt.setObjectName("subtitle")
        pl.addWidget(pt)

        self.lbl_pid  = QtWidgets.QLabel("—")
        self.lbl_pid.setObjectName("kpi_value")
        pl.addWidget(self.lbl_pid)

        self.lbl_ptype = QtWidgets.QLabel("—")
        self.lbl_ptype.setObjectName("section_label")
        pl.addWidget(self.lbl_ptype)

        self.lbl_metaphor = QtWidgets.QLabel("—")
        self.lbl_metaphor.setStyleSheet("color: #ff9f0a; font-style: italic; font-size: 12px; line-height: 14px;")
        self.lbl_metaphor.setWordWrap(True)
        pl.addWidget(self.lbl_metaphor)

        # LMA
        lma_block = QtWidgets.QGridLayout()
        lma_block.setSpacing(8)
        self.bars_calm   = {}
        self.bars_active = {}
        h1 = QtWidgets.QLabel("CALM")
        h1.setObjectName("kpi_label")
        h1.setStyleSheet("color: #64d2ff; font-weight: bold;")
        h2 = QtWidgets.QLabel("ACTIVE")
        h2.setObjectName("kpi_label")
        h2.setStyleSheet("color: #30d158; font-weight: bold;")
        lma_block.addWidget(QtWidgets.QLabel(""),  0, 0)
        lma_block.addWidget(h1, 0, 1)
        lma_block.addWidget(h2, 0, 2)

        for i, (k, label) in enumerate([("weight","Weight"),("time","Time"),("flow","Flow"),("space","Space")]):
            row = i + 1
            lbl = QtWidgets.QLabel(label)
            lbl.setObjectName("kpi_label")
            lma_block.addWidget(lbl, row, 0)
            bc = QtWidgets.QProgressBar(); bc.setRange(0,100); bc.setValue(0)
            ba = QtWidgets.QProgressBar(); ba.setRange(0,100); ba.setValue(0)
            ba.setProperty("active","true")
            self.bars_calm[k]   = bc
            self.bars_active[k] = ba
            lma_block.addWidget(bc, row, 1)
            lma_block.addWidget(ba, row, 2)
        pl.addLayout(lma_block)
        right.addWidget(p_card)

        # Tuner Profile Loader
        load_card = QtWidgets.QFrame()
        load_card.setObjectName("card")
        ll = QtWidgets.QVBoxLayout(load_card)
        ll.setContentsMargins(16, 14, 16, 16)
        ll.setSpacing(8)
        lt = QtWidgets.QLabel("PROFILE INJECTOR (TUNER JSON)")
        lt.setObjectName("subtitle")
        ll.addWidget(lt)

        id_row = QtWidgets.QHBoxLayout()
        id_row.setSpacing(8)
        self.le_pid = QtWidgets.QLineEdit()
        self.le_pid.setPlaceholderText("ID (Auto-detect)")
        id_row.addWidget(self.le_pid, stretch=1)
        self.cb_ptype = QtWidgets.QComboBox()
        self.cb_ptype.addItem("Auto-detect Profile")
        for k in PROFILE_PRESETS: self.cb_ptype.addItem(k)
        id_row.addWidget(self.cb_ptype, stretch=1)
        ll.addLayout(id_row)

        self.te_paste = QtWidgets.QPlainTextEdit()
        self.te_paste.setPlaceholderText("Paste system JSON matrix here...")
        self.te_paste.setMaximumHeight(100)
        ll.addWidget(self.te_paste)

        btn_load_row = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton("Inject Profile Matrix")
        self.btn_load.setObjectName("primary")
        self.btn_load.clicked.connect(self._load_profile_clicked)
        btn_load_row.addStretch()
        btn_load_row.addWidget(self.btn_load)
        ll.addLayout(btn_load_row)

        self.lbl_load_status = QtWidgets.QLabel("")
        self.lbl_load_status.setStyleSheet("color: #8e8e93; font-size: 11px;")
        self.lbl_load_status.setWordWrap(True)
        ll.addWidget(self.lbl_load_status)
        right.addWidget(load_card)

        # Status KPIs
        s_card = QtWidgets.QFrame()
        s_card.setObjectName("card")
        sl = QtWidgets.QVBoxLayout(s_card)
        sl.setContentsMargins(16, 14, 16, 16)
        sl.setSpacing(8)
        st = QtWidgets.QLabel("SYSTEM LIVE METRICS")
        st.setObjectName("subtitle")
        sl.addWidget(st)

        kpi_row = QtWidgets.QHBoxLayout()
        kpi_row.setSpacing(12)
        self.lbl_kpi_action  = self._make_kpi("CURRENT BEHAVIOR", "—")
        self.lbl_kpi_posture = self._make_kpi("HUMAN POSTURE", "—")
        self.lbl_kpi_elapsed = self._make_kpi("ELAPSED TIME", "—")
        kpi_row.addLayout(self.lbl_kpi_action)
        kpi_row.addLayout(self.lbl_kpi_posture)
        kpi_row.addLayout(self.lbl_kpi_elapsed)
        sl.addLayout(kpi_row)
        right.addWidget(s_card)

        right.addStretch()

        # Footer
        footer = QtWidgets.QLabel("H: Toggle Inspector  •  Q: Safe Exit")
        footer.setStyleSheet("color: #48484a; font-size: 11px; font-weight: bold;")
        footer.setAlignment(QtCore.Qt.AlignCenter)
        right.addWidget(footer)

        root.addLayout(right, stretch=3)

    # ───────────────────────────────────────────────────────────────────────────
    def _make_reasoning_row(self, label):
        row = QtWidgets.QVBoxLayout()
        row.setSpacing(4)
        l = QtWidgets.QLabel(label)
        l.setObjectName("kpi_label")
        v = QtWidgets.QLabel("—")
        v.setWordWrap(True)
        v.setStyleSheet("color: #d1d1d6; font-size: 13px; background-color: #121214; padding: 6px 10px; border-radius: 6px; border: 1px solid #2d2d30;")
        row.addWidget(l)
        row.addWidget(v)
        row._label = l
        row._value = v
        return row

    def _make_kpi(self, label, value):
        v = QtWidgets.QVBoxLayout()
        v.setSpacing(4)
        lv = QtWidgets.QLabel(value)
        lv.setObjectName("kpi_value")
        # fix: minimum width for the KPI counters so changing digits do not shrink the whole layout
        lv.setMinimumWidth(130)
        lv.setAlignment(QtCore.Qt.AlignLeft)
        ll = QtWidgets.QLabel(label)
        ll.setObjectName("kpi_label")
        v.addWidget(lv)
        v.addWidget(ll)
        v._value = lv
        return v

    # ───────────────────────────────────────────────────────────────────────────
    def _wire_ros(self):
        if not self.ros: return
        self.ros.ext_frame_received.connect(self.cam_ext.update_frame)
        self.ros.eye_frame_received.connect(self.cam_eye.update_frame)
        self.ros.ui_data_received.connect(self._on_ui_data)

    # ───────────────────────────────────────────────────────────────────────────
    def _on_ui_data(self, d):
        if d.get("participant"): self.lbl_pid.setText(str(d["participant"]))
        if d.get("profile_type"): self.lbl_ptype.setText(str(d["profile_type"]))
        if d.get("metaphor"): self.lbl_metaphor.setText(str(d["metaphor"]) or "—")

        lc = d.get("lma_calm", {}) or {}
        la = d.get("lma_active", {}) or {}
        for k in ("weight","time","flow","space"):
            self.bars_calm[k].setValue(int(lc.get(k, 0)))
            self.bars_active[k].setValue(int(la.get(k, 0)))

        if self.inspector:
            self.lbl_visual._value.setText(d.get("visual") or "—")
            self.lbl_thought._value.setText(d.get("thought") or "—")
            self.lbl_reason._value.setText(d.get("reason") or "—")

        self.lbl_kpi_action._value.setText((d.get("current_action") or "—").upper().replace("_", " "))
        self.lbl_kpi_posture._value.setText((d.get("posture") or d.get("state") or "—").upper())
        if d.get("elapsed"): self.lbl_kpi_elapsed._value.setText(d["elapsed"])

    # ───────────────────────────────────────────────────────────────────────────
    def _switch_mode(self, mode):
        ok, msg = self.pm.start(mode)
        print(f"[Mode] {msg}")
        self._update_mode_buttons()

    def _stop_mode(self):
        self.pm.stop()
        self._update_mode_buttons()

    def _update_mode_buttons(self):
        m = self.pm.mode if self.pm.is_running() else None
        for btn, mode in ((self.btn_spont, "spontaneous"), (self.btn_mech, "mechanical")):
            active = (m == mode) and self.inspector
            btn.setProperty("active", "true" if active else "false")
            btn.style().unpolish(btn); btn.style().polish(btn)

        if not self.pm.is_running():
            self.status_pill.setText("STOPPED")
            self.status_pill.setProperty("state", "stopped")
        elif self.inspector:
            if m == "spontaneous":
                self.status_pill.setText("● SPONTANEOUS")
                self.status_pill.setProperty("state", "running_spont")
            else:
                self.status_pill.setText("● MECHANICAL")
                self.status_pill.setProperty("state", "running_mech")
        else:
            self.status_pill.setText("● RUNNING")
            self.status_pill.setProperty("state", "running_spont")
        self.status_pill.style().unpolish(self.status_pill); self.status_pill.style().polish(self.status_pill)

    def _heartbeat(self):
        self._update_mode_buttons()

    def _load_profile_clicked(self):
        text = self.te_paste.toPlainText()
        pid  = self.le_pid.text().strip() or None
        ptype_sel = self.cb_ptype.currentText()
        override = None if ptype_sel == "Auto-detect Profile" else ptype_sel

        ok, result = write_profile_from_tuner(text, pid, override)
        if ok:
            self.lbl_load_status.setStyleSheet("color: #30d158; font-size: 12px;")
            self.lbl_load_status.setText(f"✓ Loaded {result['participantId']} ({result['profileType']}).")
            self._refresh_profile_display()
        else:
            self.lbl_load_status.setStyleSheet("color: #ff453a; font-size: 12px;")
            self.lbl_load_status.setText(f"✗ {result}")

    def _refresh_profile_display(self):
        try:
            with open(PROFILE_PATH, encoding="utf-8") as f:
                p = json.load(f)
            self.lbl_pid.setText(p.get("participantId", "—"))
            self.lbl_ptype.setText(p.get("profileType", "—"))
            self.lbl_metaphor.setText(p.get("mentalMetaphor","—") or "—")
            lc = p.get("lma_calm", {}) or {}
            la = p.get("lma_active", {}) or {}
            for k in ("weight","time","flow","space"):
                self.bars_calm[k].setValue(int(lc.get(k, 0)))
                self.bars_active[k].setValue(int(la.get(k, 0)))
        except Exception: pass

    def keyPressEvent(self, ev):
        k = ev.key()
        if k == QtCore.Qt.Key_H:
            self.inspector = not self.inspector
            self.inspector_badge.setVisible(self.inspector)
            for row in (self.lbl_visual, self.lbl_thought, self.lbl_reason):
                row._label.setVisible(self.inspector)
                row._value.setVisible(self.inspector)
            self._update_mode_buttons()
        elif k == QtCore.Qt.Key_Q: self.close()
        elif k == QtCore.Qt.Key_S: self._switch_mode("spontaneous")
        elif k == QtCore.Qt.Key_M: self._switch_mode("mechanical")
        elif k == QtCore.Qt.Key_X: self._stop_mode()
        super().keyPressEvent(ev)

    def closeEvent(self, ev):
        self.pm.stop()
        super().closeEvent(ev)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

    ros_bridge = RosBridge() if HAS_ROS else None
    win = MainWindow(ros_bridge)
    win.show()

    if HAS_ROS:
        t = threading.Thread(target=lambda: rospy.spin(), daemon=True)
        t.start()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
