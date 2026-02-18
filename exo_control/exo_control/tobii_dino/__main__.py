#!/usr/bin/env python3
"""
G3 Tobii Glasses 3 Live App (Kivy) + Open-Vocabulary Detector (GroundingDINO)
+ Hand-overlap grasp state (MediaPipe Hands)
+ Parallel VLM (Phi-3.5-Vision-Instruct) for weight estimation
Trigger rule for VLM: NEW OBJECT = NEW LABEL ONLY (first time label appears in scene)

Targets (open-vocab):
- Cardboard Box
- Screwdriver
- Drill
- Screw

For each detection we derive Grasped / Not-Grasped via hands overlap.
"""

import asyncio
import json
import logging
import re
import time
import traceback
from typing import Dict, List, Optional, Set, Tuple, cast

import aiohttp
import numpy as np
from eventkinds import AppEventKind, ControlEventKind
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Line, Rectangle
from kivy.graphics.texture import Texture
from kivy.lang.builder import Builder
from kivy.metrics import dp
from kivy.properties import BooleanProperty
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.recycleview import RecycleView
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.screenmanager import Screen, ScreenManager

from g3pylib import Glasses3, connect_to_glasses
from g3pylib.g3typing import SignalBody
from g3pylib.recordings import RecordingsEventKind
from g3pylib.recordings.recording import Recording
from g3pylib.zeroconf import EventKind, G3Service, G3ServiceDiscovery

# OpenCV
import cv2
from PIL import Image
from collections import deque

# ROS2
import rclpy
from rclpy.node import Node
from exo_interfaces.srv import SetAdmittanceParams
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState

# ====================== OPTIONAL DEPENDENCIES ======================
# GroundingDINO (open-vocabulary detection)
# pip install groundingdino-py
DINO_AVAILABLE = True
try:
    from groundingdino.util.inference import load_model as dino_load_model
    from groundingdino.util.inference import predict as dino_predict
    from groundingdino.datasets import transforms as dino_T
except Exception as e:
    DINO_AVAILABLE = False
    print(f"[WARN] GroundingDINO not available: {e}")

# MediaPipe Hands (fast grasp-state proxy)
# pip install mediapipe
MP_AVAILABLE = True
try:
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        raise ImportError("mediapipe.solutions not available (new tasks-only API)")
except Exception as e:
    MP_AVAILABLE = False
    print(f"[WARN] mediapipe not available: {e}")

# Local VLM (Phi-3.5 vision) via transformers + bitsandbytes
# pip install transformers accelerate bitsandbytes
VLM_AVAILABLE = True
try:
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
except Exception as e:
    VLM_AVAILABLE = False
    print(f"[WARN] VLM dependencies not available: {e}")

# ====================== PARAMS ======================

# Video/Gaze
GAZE_CIRCLE_RADIUS = 10
VIDEOPLAYER_PROGRESS_BAR_HEIGHT = dp(44)
VIDEO_Y_TO_X_RATIO = 9 / 16
LIVE_FRAME_RATE = 25

# Detector (high frequency)
DETECTOR_RATE_HZ = 8.0  # you can change this
SHOW_OVERLAY = True

# Open-vocab classes (canonical names you want)
TARGET_LABELS = ["Cardboard Box", "Screwdriver", "Drill", "Screw"]
# Prompt text for DINO
DINO_PROMPT = "cardboard box . screwdriver . drill . screw ."

# DINO thresholds
DINO_BOX_THRESH = 0.35
DINO_TEXT_THRESH = 0.25

# Hand overlap grasp heuristic
# Grasped if intersection_area / object_area >= this
GRASP_INTERSECTION_RATIO = 0.08

# VLM (weight estimation on new label only)
VLM_MODEL_ID = "microsoft/Phi-3.5-vision-instruct"
VLM_MAX_NEW_TOKENS = 128

# Box gate debounce & heartbeat (kept as in your script)
BOX_GATE_HEARTBEAT_HZ = 5.0

BOX_GATE_ON_WIN = 6
BOX_GATE_ON_RATIO = 0.6
BOX_GATE_ON_CONSEC = 0

BOX_GATE_OFF_WIN = 8
BOX_GATE_OFF_RATIO = 0.6
BOX_GATE_OFF_CONSEC = 0

# Which object drives your ROS "box_gate"?
# Keep semantics close to your previous logic: gate follows "Cardboard Box" grasp state.
GATE_LABEL_GRASPED = "Cardboard Box-Grasped"
GATE_LABEL_NOT_GRASPED = "Cardboard Box-Not Grasped"

# ROS2 service names & admittance
SET_ADMITT_SRV_NAME = "set_admittance_params"
K_DOWN = 30.0
K_UP = 30.0

# Place via thresholds relative JointState (kept)
DELTA_DOWN_PLACE = 0.04
DELTA_UP_PLACE = 0.05
ARM_DELAY_S = 2.5
JOINT_NAME = "joint_0"
FALLBACK_INDEX = 0

logging.basicConfig(level=logging.DEBUG)

# ====================== KIVY KV ======================
Builder.load_string(
    """
#:import NoTransition kivy.uix.screenmanager.NoTransition
#:import Factory kivy.factory.Factory
#:import ControlEventKind eventkinds.ControlEventKind
#:import AppEventKind eventkinds.AppEventKind

<DiscoveryScreen>:
    BoxLayout:
        BoxLayout:
            orientation: "vertical"
            Label:
                size_hint_y: None
                height: dp(50)
                text: "Found services:"
            SelectableList:
                id: services
        Button:
            size_hint: 1, None
            height: dp(50)
            pos_hint: {'center_x':0.5, 'center_y':0.5}
            text: "Connect"
            on_press: app.send_app_event(AppEventKind.ENTER_CONTROL_SESSION)

<UserMessagePopup>:
    size_hint: None, None
    size: 400, 200
    Label:
        id: message_label
        text: ""

<ControlScreen>:
    BoxLayout:
        orientation: 'vertical'
        BoxLayout:
            size_hint: 1, None
            height: dp(50)
            Label:
                id: hostname
                text: "Hostname placeholder"
                halign: "left"
            Label:
                id: task_indicator
                text: ""
        BoxLayout:
            size_hint: 1, None
            height: dp(50)
            Button:
                text: "Recorder"
                on_press: root.switch_to_screen("recorder")
            Button:
                text: "Live"
                on_press: root.switch_to_screen("live")
            Button:
                background_color: (0.6, 0.6, 1, 1)
                text: "Disconnect"
                on_press:
                    app.send_app_event(AppEventKind.LEAVE_CONTROL_SESSION)
        ScreenManager:
            id: sm
            transition: NoTransition()

<RecordingScreen>:
    VideoPlayer:
        id: videoplayer

<RecorderScreen>:
    BoxLayout:
        BoxLayout:
            orientation: 'vertical'
            Label:
                id: recorder_status
                text: "Status:"
            Button:
                text: "Start"
                on_press: app.send_control_event(ControlEventKind.START_RECORDING)
            Button:
                text: "Stop"
                on_press: app.send_control_event(ControlEventKind.STOP_RECORDING)
            Button:
                text: "Delete"
                on_press: app.send_control_event(ControlEventKind.DELETE_RECORDING)
            Button:
                text: "Play"
                on_press: app.send_control_event(ControlEventKind.PLAY_RECORDING)
        SelectableList:
            id: recordings

<LiveScreen>:
    BoxLayout:
        Widget:
            id: display
            size_hint_x: 0.8
            size_hint_y: 1
        BoxLayout:
            orientation: "vertical"
            size_hint_x: 0.2
            Button:
                text: "Start"
                on_press: app.send_control_event(ControlEventKind.START_LIVE)
            Button:
                text: "Stop"
                on_press: app.send_control_event(ControlEventKind.STOP_LIVE)

<SelectableList>:
    viewclass: 'SelectableLabel'
    SelectableRecycleBoxLayout:
        id: selectables
        default_size: None, dp(70)
        default_size_hint: 1, None
        size_hint_y: None
        height: self.minimum_height
        orientation: 'vertical'

<SelectableLabel>:
    canvas.before:
        Color:
            rgba: (.0, 0.9, .1, .3) if self.selected else (0, 0, 0, 1)
        Rectangle:
            pos: self.pos
            size: self.size
"""
)

# ====================== UI CLASSES ======================
class SelectableRecycleBoxLayout(LayoutSelectionBehavior, RecycleBoxLayout):
    pass


class SelectableLabel(RecycleDataViewBehavior, Label):
    index = None
    selected = BooleanProperty(False)
    selectable = BooleanProperty(True)

    def refresh_view_attrs(self, rv, index, data):
        self.index = index
        return super().refresh_view_attrs(rv, index, data)

    def on_touch_down(self, touch):
        if super().on_touch_down(touch):
            return True
        if self.collide_point(*touch.pos) and self.selectable:
            return self.parent.select_with_touch(self.index, touch)

    def apply_selection(self, rv, index, is_selected):
        self.selected = is_selected


class SelectableList(RecycleView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = []


class DiscoveryScreen(Screen):
    def add_service(self, hostname: str, ipv4: Optional[str], ipv6: Optional[str]) -> None:
        self.ids.services.data.append({"hostname": hostname, "text": f"{hostname}\n{ipv4}\n{ipv6}"})
        logging.info(f"Services: Added {hostname}, {ipv4}, {ipv6}")

    def update_service(self, hostname: str, ipv4: Optional[str], ipv6: Optional[str]) -> None:
        services = self.ids.services
        for service in services.data:
            if service["hostname"] == hostname:
                service["text"] = f"{hostname}\n{ipv4}\n{ipv6}"
                logging.info(f"Services: Updated {hostname}, {ipv4}, {ipv6}")

    def remove_service(self, hostname: str, ipv4: Optional[str], ipv6: Optional[str]) -> None:
        services = self.ids.services
        services.data = [service for service in services.data if service["hostname"] != hostname]
        logging.info(f"Services: Removed {hostname}, {ipv4}, {ipv6}")

    def clear(self):
        self.ids.services.data = []
        logging.info("Services: All cleared")


class ControlScreen(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.ids.sm.add_widget(RecorderScreen(name="recorder"))
        self.ids.sm.add_widget(RecordingScreen(name="recording"))
        self.ids.sm.add_widget(LiveScreen(name="live"))

    def clear(self) -> None:
        self.ids.sm.get_screen("recorder").ids.recordings.data = []
        self.ids.sm.get_screen("recorder").ids.recorder_status.text = "Status:"

    def switch_to_screen(self, screen: str) -> None:
        self.ids.sm.current = screen
        if self.ids.sm.current == "recording":
            self.ids.sm.get_screen("recording").ids.videoplayer.state = "stop"

    def set_task_running_status(self, is_running: bool) -> None:
        self.ids.task_indicator.text = "Handling action..." if is_running else ""

    def set_hostname(self, hostname: str) -> None:
        self.ids.hostname.text = hostname


class RecordingScreen(Screen):
    pass


class RecorderScreen(Screen):
    def add_recording(self, visible_name: str, uuid: str, recording: Recording, atEnd: bool = False) -> None:
        recordings = self.ids.recordings
        data = {"text": visible_name, "uuid": uuid, "recording": recording}
        (recordings.data.append if atEnd else recordings.data.insert)(0, data)

    def remove_recording(self, uuid: str) -> None:
        recordings = self.ids.recordings
        recordings.data = [rec for rec in recordings.data if rec["uuid"] != uuid]

    def set_recording_status(self, is_recording: bool) -> None:
        self.ids.recorder_status.text = "Status: Recording" if is_recording else "Status: Not recording"


class LiveScreen(Screen):
    def clear(self, *args):
        self.ids.display.canvas.clear()


class UserMessagePopup(Popup):
    pass


class GazeCircle:
    def __init__(self, canvas, origin, size) -> None:
        self.canvas = canvas
        self.origin = origin
        self.size = size
        self.circle_obj = Line(circle=(0, 0, 0))
        self.canvas.add(self.circle_obj)

    def redraw(self, coord):
        self.canvas.remove(self.circle_obj)
        self.canvas.add(Color(1, 0, 0, 1))
        if coord is None:
            self.circle_obj = Line(circle=(0, 0, 0))
        else:
            cx = self.origin[0] + coord[0] * self.size[0]
            cy = self.origin[1] + (1 - coord[1]) * self.size[1]
            self.circle_obj = Line(circle=(cx, cy, GAZE_CIRCLE_RADIUS))
        self.canvas.add(self.circle_obj)
        self.canvas.remove(Color(1, 0, 0, 1))

    def reset(self):
        self.canvas.remove(self.circle_obj)
        self.circle_obj = Line(circle=(0, 0, 0))
        self.canvas.add(self.circle_obj)


# ====================== APP ======================
class G3App(App, ScreenManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_request_close=self.close)

        # Core app queues/tasks
        self.tasks: Set[asyncio.Task] = set()
        self.app_events: asyncio.Queue[AppEventKind] = asyncio.Queue()
        self.control_events: asyncio.Queue[ControlEventKind] = asyncio.Queue()
        self.live_stream_task: Optional[asyncio.Task] = None
        self.read_frames_task: Optional[asyncio.Task] = None

        self.add_widget(DiscoveryScreen(name="discovery"))
        self.add_widget(ControlScreen(name="control"))

        # Latest streams
        self.latest_frame_with_timestamp = None
        self.latest_gaze_with_timestamp = None

        # Rendering
        self.live_gaze_circle = None
        self.replay_gaze_circle = None
        self.last_texture = None
        self.draw_frame_event = None
        self._last_gaze_print = 0.0

        # ======================
        # Perception state
        # ======================
        self._detector_task: Optional[asyncio.Task] = None
        self._detections_latest: List[dict] = []  # [{"xyxy":[...], "label":..., "label_state":..., "conf":..., "grasped":...}, ...]
        self._last_dino_log_ts = 0.0

        # New label trigger for VLM weight
        self._labels_seen: Set[str] = set()
        self._weight_by_label: Dict[str, dict] = {}  # label -> weight json dict
        self._vlm_queue: asyncio.Queue[Tuple[str, np.ndarray]] = asyncio.Queue()
        self._vlm_task: Optional[asyncio.Task] = None

        # ======================
        # GroundingDINO
        # ======================
        self.dino_model = None
        # Set these paths to your local GroundingDINO config/weights if needed
        # You can download official weights from the GroundingDINO repo releases.
        self.DINO_CONFIG_PATH = "/home/alessandro/models/groundingdino/GroundingDINO_SwinT_OGC.py"
        self.DINO_WEIGHTS_PATH = "/home/alessandro/models/groundingdino/groundingdino_swint_ogc.pth"

        # ======================
        # MediaPipe Hands
        # ======================
        self.mp_hands = None
        self.hands_detector = None

        # ======================
        # VLM (Phi-3.5 Vision)
        # ======================
        self.vlm_processor = None
        self.vlm_model = None

        # ======================
        # ROS
        # ======================
        self.ros_node: Node | None = None
        self.adm_client = None
        self.box_gate_pub = None
        self.js_sub = None
        self._ros_ready = False

        # Box gate debounce
        self.box_gate = False
        self._on_dwell = deque(maxlen=BOX_GATE_ON_WIN)
        self._off_dwell = deque(maxlen=BOX_GATE_OFF_WIN)
        self._gate_heartbeat_task = None

        # Place baselines (kept)
        self.state = "stand_no_box"  # 'stand_with_box', 'bend_to_place'
        self.jidx = None
        self.last_pos = None
        self.baseline_swB = None
        self.baseline_b2p = None
        self.arm_until = 0.0

    # ---------- GAZE in bbox ----------
    def _gaze_in_label_box(self, det_list, frame_w: int, frame_h: int, wanted_label: str) -> bool:
        """
        True if gaze falls inside any bbox whose label_state == wanted_label.
        NOTE: We check d["label_state"] first, fallback to d["label"].
        """
        if self.latest_gaze_with_timestamp is None:
            return False
        gaze_packet = self.latest_gaze_with_timestamp[0]
        if not gaze_packet or len(gaze_packet) == 0:
            return False
        try:
            gx, gy = gaze_packet["gaze2d"]
        except Exception:
            return False

        gx = min(max(float(gx), 0.0), 1.0)
        gy = min(max(float(gy), 0.0), 1.0)
        x_px = gx * frame_w
        y_px = (1.0 - gy) * frame_h  # frame is vertically flipped before display

        for d in det_list:
            lab = d.get("label_state", d.get("label", ""))
            if lab != wanted_label:
                continue
            x1, y1, x2, y2 = d["xyxy"]
            if x1 <= x_px <= x2 and y1 <= y_px <= y2:
                return True
        return False

    # ---------- ROS ----------
    def _init_ros(self):
        if self._ros_ready:
            return
        if not rclpy.ok():
            rclpy.init()
        self.ros_node = rclpy.create_node("vision_command_node")
        self.adm_client = self.ros_node.create_client(SetAdmittanceParams, SET_ADMITT_SRV_NAME)
        self.box_gate_pub = self.ros_node.create_publisher(Bool, "perception/box_gate", 10)
        self.js_sub = self.ros_node.create_subscription(JointState, "joint_states", self._on_js, 30)

        self.box_gate_pub.publish(Bool(data=False))
        self.box_gate = False

        if not self.adm_client.wait_for_service(timeout_sec=5.0):
            print(f"[ROS] ERROR: service '{SET_ADMITT_SRV_NAME}' not available")

        self._ros_ready = True
        if self._gate_heartbeat_task is None or self._gate_heartbeat_task.done():
            self._gate_heartbeat_task = self.create_task(self._box_gate_heartbeat_loop(), name="box_gate_heartbeat")

        print("[ROS] Clients ready.")

    def _shutdown_ros(self):
        try:
            if self.ros_node is not None:
                self.ros_node.destroy_node()
                self.ros_node = None
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as e:
            print(f"[ROS] shutdown error: {e}")
        finally:
            self._ros_ready = False
            self.adm_client = None
            self.box_gate_pub = None
            self.js_sub = None
        self._gate_heartbeat_task = None

    def _set_admittance_params_blocking(self, k_val: float):
        if not self._ros_ready or self.ros_node is None or self.adm_client is None:
            print("[ROS] adm client not ready")
            return
        req = SetAdmittanceParams.Request()
        req.k = float(k_val)
        future = self.adm_client.call_async(req)
        rclpy.spin_until_future_complete(self.ros_node, future)

    async def _trigger_action_ros(self, action: str):
        if not self._ros_ready:
            print("[ROS] not ready, skipping action")
            return
        k_val = K_DOWN if action == "down" else K_UP
        await asyncio.to_thread(self._set_admittance_params_blocking, k_val)

    def _set_box_gate(self, val: bool):
        if not self._ros_ready:
            self.box_gate = val
            return
        if val != self.box_gate:
            self.box_gate = val
            self.box_gate_pub.publish(Bool(data=val))
            print(f"[BOX_GATE] edge → {val}")

    async def _box_gate_heartbeat_loop(self):
        period = 1.0 / max(0.1, BOX_GATE_HEARTBEAT_HZ)
        while self._ros_ready:
            try:
                if self.box_gate_pub is not None:
                    self.box_gate_pub.publish(Bool(data=self.box_gate))
                await asyncio.sleep(period)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[BOX_GATE] heartbeat error: {e}")
                await asyncio.sleep(period)

    # ---------- JointState → place (kept) ----------
    def _on_js(self, msg: JointState):
        if self.jidx is None:
            if msg.name and JOINT_NAME in msg.name:
                self.jidx = msg.name.index(JOINT_NAME)
            else:
                self.jidx = min(FALLBACK_INDEX, len(msg.position) - 1)
            print(
                f"[MEAS READY] Using joint index {self.jidx} "
                f"({'name:'+JOINT_NAME if msg.name else 'no names'})"
            )
            return
        if self.jidx >= len(msg.position):
            return

        pos = float(msg.position[self.jidx])
        self.last_pos = pos
        now = time.monotonic()

        if self.state == "stand_with_box":
            if self.baseline_swB is None:
                if now < self.arm_until:
                    return
                self.baseline_swB = pos
                return
            delta = pos - self.baseline_swB
            if delta >= DELTA_DOWN_PLACE:
                self.state = "bend_to_place"
                self.baseline_b2p = None
                self.arm_until = now + ARM_DELAY_S
                self.create_task(self._trigger_action_ros("down"), name=f"ros_action_down_{int(now)}")
                return

        elif self.state == "bend_to_place":
            if self.baseline_b2p is None:
                if now < self.arm_until:
                    return
                self.baseline_b2p = pos
                return
            delta = self.baseline_b2p - pos
            if delta >= DELTA_UP_PLACE:
                self.state = "stand_no_box"
                self._set_box_gate(False)
                self.baseline_swB = None
                self.baseline_b2p = None
                self.arm_until = now + ARM_DELAY_S
                self.create_task(self._trigger_action_ros("up"), name=f"ros_action_up_{int(now)}")
                return

    # ======================
    # Perception: DINO + Hands
    # ======================
    def _ensure_dino(self):
        if not DINO_AVAILABLE:
            raise RuntimeError("GroundingDINO not installed/importable.")
        if self.dino_model is None:
            print("[DINO] Loading model...")
            self.dino_model = dino_load_model(self.DINO_CONFIG_PATH, self.DINO_WEIGHTS_PATH)
            print("[DINO] Ready.")

    def _ensure_hands(self):
        if not MP_AVAILABLE:
            raise RuntimeError("mediapipe not installed/importable.")
        if self.hands_detector is None:
            self.mp_hands = mp.solutions.hands
            # tune for speed; you can raise max_num_hands if needed
            self.hands_detector = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            print("[HANDS] Ready.")

    @staticmethod
    def _canonicalize_label(raw: str) -> Optional[str]:
        """Map DINO raw labels to your canonical TARGET_LABELS."""
        s = raw.strip().lower()
        if "cardboard" in s and "box" in s:
            return "Cardboard Box"
        if "screwdriver" in s:
            return "Screwdriver"
        if "drill" in s:
            return "Drill"
        if s == "screw" or "screw " in s or " screw" in s:
            return "Screw"
        return None

    @staticmethod
    def _clamp_xyxy(xyxy, w, h):
        x1, y1, x2, y2 = map(int, xyxy)
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    @staticmethod
    def _intersection_ratio(obj_xyxy, hand_xyxy) -> float:
        x1, y1, x2, y2 = obj_xyxy
        hx1, hy1, hx2, hy2 = hand_xyxy
        ix1 = max(x1, hx1)
        iy1 = max(y1, hy1)
        ix2 = min(x2, hx2)
        iy2 = min(y2, hy2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        obj_area = max(1, (x2 - x1) * (y2 - y1))
        return inter / obj_area

    def _detect_hands_xyxy(self, frame_bgr: np.ndarray) -> List[List[int]]:
        """
        Returns list of hand bboxes in pixel coords: [[x1,y1,x2,y2], ...]
        """
        self._ensure_hands()
        h, w = frame_bgr.shape[:2]
        # mediapipe expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands_detector.process(frame_rgb)
        out = []
        if not res.multi_hand_landmarks:
            return out
        for hand_lms in res.multi_hand_landmarks:
            xs = [lm.x for lm in hand_lms.landmark]
            ys = [lm.y for lm in hand_lms.landmark]
            x1 = int(max(0, min(xs) * w))
            x2 = int(min(w - 1, max(xs) * w))
            y1 = int(max(0, min(ys) * h))
            y2 = int(min(h - 1, max(ys) * h))
            if x2 > x1 and y2 > y1:
                out.append([x1, y1, x2, y2])
        return out

    def _run_dino_blocking(self, frame_bgr: np.ndarray) -> List[dict]:
        """
        GroundingDINO detection (blocking).
        Returns det_list with canonical labels, without grasp info yet.
        """
        self._ensure_dino()
        h, w = frame_bgr.shape[:2]

        # DINO utilities expect RGB and a normalized torch tensor
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        dino_transform = dino_T.Compose(
            [
                dino_T.RandomResize([800], max_size=1333),
                dino_T.ToTensor(),
                dino_T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = dino_transform(image_pil, None)
        # Align input dtype/device with model (avoid Float vs Half matmul errors)
        try:
            p0 = next(self.dino_model.parameters())
            image = image.to(device=p0.device, dtype=p0.dtype)
        except Exception:
            pass

        boxes, logits, phrases = dino_predict(
            model=self.dino_model,
            image=image,
            caption=DINO_PROMPT,
            box_threshold=DINO_BOX_THRESH,
            text_threshold=DINO_TEXT_THRESH,
        )

        dets: List[dict] = []
        # boxes are in xyxy normalized? depending on util; groundingdino.util.inference returns boxes in (x1,y1,x2,y2) normalized [0,1]
        # We'll handle both normalized and absolute robustly:
        for b, score, phr in zip(boxes, logits, phrases):
            raw_label = str(phr)
            canon = self._canonicalize_label(raw_label)
            if canon is None or canon not in TARGET_LABELS:
                continue

            bb = b.tolist() if hasattr(b, "tolist") else list(b)
            # detect normalized
            if max(bb) <= 1.5:
                x1 = int(bb[0] * w)
                y1 = int(bb[1] * h)
                x2 = int(bb[2] * w)
                y2 = int(bb[3] * h)
                xyxy = [x1, y1, x2, y2]
            else:
                xyxy = [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])]

            xyxy = self._clamp_xyxy(xyxy, w, h)
            if xyxy is None:
                continue

            dets.append(
                {
                    "xyxy": xyxy,
                    "label": canon,
                    "conf": float(score) if score is not None else 0.0,
                }
            )

        # sort by confidence
        dets.sort(key=lambda d: d["conf"], reverse=True)
        return dets

    def _add_grasp_state(self, dets: List[dict], hand_boxes: List[List[int]]) -> List[dict]:
        """
        Adds:
          - grasped: bool
          - label_state: "<Label>-Grasped" / "<Label>-Not Grasped"
        """
        out = []
        for d in dets:
            obj_xyxy = d["xyxy"]
            grasped = False
            for hb in hand_boxes:
                if self._intersection_ratio(obj_xyxy, hb) >= GRASP_INTERSECTION_RATIO:
                    grasped = True
                    break
            d2 = dict(d)
            d2["grasped"] = grasped
            d2["label_state"] = f"{d2['label']}-Grasped" if grasped else f"{d2['label']}-Not Grasped"
            out.append(d2)
        return out

    async def _detector_loop(self, rate_hz: float = DETECTOR_RATE_HZ):
        """
        High-rate detector loop:
          - reads latest frame
          - DINO boxes (open-vocab)
          - hand detection
          - derives grasp state
          - updates self._detections_latest
          - updates ROS box_gate using gaze dwell on Cardboard Box grasp state
          - enqueues VLM weight job ONCE per new label
        """
        period = 1.0 / max(0.5, float(rate_hz))
        while True:
            start = time.monotonic()
            try:
                fw = self.latest_frame_with_timestamp
                if not fw or fw[0] is None:
                    await asyncio.sleep(0.01)
                    continue

                # frame BGR flipped vertically (consistent with draw)
                frame = np.flip(fw[0].to_ndarray(format="bgr24"), 0)
                h, w = frame.shape[:2]

                # DINO inference in thread
                dets = await asyncio.to_thread(self._run_dino_blocking, frame)

                # hands in thread (mediapipe can be a bit heavy)
                hand_boxes = await asyncio.to_thread(self._detect_hands_xyxy, frame) if MP_AVAILABLE else []

                dets = self._add_grasp_state(dets, hand_boxes)

                # Update shared detections for overlay
                self._detections_latest = dets
                now = time.monotonic()
                if now - self._last_dino_log_ts >= 1.0:
                    items = [
                        {
                            "label": d.get("label", ""),
                            "xyxy": d.get("xyxy", []),
                            "conf": d.get("conf", None),
                        }
                        for d in dets
                    ]
                    print(f"[DINO] detections={len(dets)} items={items}")
                    self._last_dino_log_ts = now

                # ---- VLM trigger: new LABEL only ----
                # Take the top confidence bbox for the new label and enqueue one job.
                for lab in TARGET_LABELS:
                    if lab in self._labels_seen:
                        continue
                    # if lab appears now:
                    cand = [d for d in dets if d.get("label") == lab]
                    if len(cand) == 0:
                        continue
                    best = cand[0]
                    x1, y1, x2, y2 = best["xyxy"]
                    crop = frame[y1:y2, x1:x2].copy()
                    self._labels_seen.add(lab)
                    # placeholder initial
                    self._weight_by_label.setdefault(lab, {"status": "queued"})
                    await self._vlm_queue.put((lab, crop))
                    print(f"[VLM] queued weight estimation for NEW label: {lab}")
                    # IMPORTANT: new label only => we do not queue again
                    # break not needed: a frame can introduce multiple new labels

                # ---- BOX GATE (dwell on cardboard box grasp state) ----
                gaze_in_grasped = self._gaze_in_label_box(dets, w, h, GATE_LABEL_GRASPED)
                gaze_in_notgrasped = self._gaze_in_label_box(dets, w, h, GATE_LABEL_NOT_GRASPED)

                if not self.box_gate:
                    self._on_dwell.append(1 if gaze_in_grasped else 0)
                    turn_on = False
                    if BOX_GATE_ON_CONSEC > 0:
                        if len(self._on_dwell) >= BOX_GATE_ON_CONSEC and all(
                            x == 1 for x in list(self._on_dwell)[-BOX_GATE_ON_CONSEC :]
                        ):
                            turn_on = True
                    else:
                        if len(self._on_dwell) >= max(1, BOX_GATE_ON_WIN):
                            ratio_on = sum(self._on_dwell) / len(self._on_dwell)
                            if ratio_on >= BOX_GATE_ON_RATIO:
                                turn_on = True
                    if turn_on:
                        self._off_dwell.clear()
                        self._on_dwell.clear()
                        self._set_box_gate(True)
                        # optional: keep your place FSM semantics if you want
                        self.state = "stand_with_box"
                        self.baseline_swB = None
                        self.baseline_b2p = None
                        self.arm_until = time.monotonic() + ARM_DELAY_S

                else:
                    self._off_dwell.append(1 if gaze_in_notgrasped else 0)
                    turn_off = False
                    if BOX_GATE_OFF_CONSEC > 0:
                        if len(self._off_dwell) >= BOX_GATE_OFF_CONSEC and all(
                            x == 1 for x in list(self._off_dwell)[-BOX_GATE_OFF_CONSEC :]
                        ):
                            turn_off = True
                    else:
                        if len(self._off_dwell) >= max(1, BOX_GATE_OFF_WIN):
                            ratio_off = sum(self._off_dwell) / len(self._off_dwell)
                            if ratio_off >= BOX_GATE_OFF_RATIO:
                                turn_off = True
                    if turn_off:
                        self._on_dwell.clear()
                        self._off_dwell.clear()
                        self._set_box_gate(False)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[DETECTOR] Error: {e}")

            # process ROS callbacks (non-blocking)
            if self._ros_ready:
                try:
                    rclpy.spin_once(self.ros_node, timeout_sec=0.0)
                except Exception:
                    pass

            dt = time.monotonic() - start
            await asyncio.sleep(max(0.0, period - dt))

    # ======================
    # VLM worker: Phi-3.5 Vision weight estimation
    # ======================
    def _ensure_vlm(self):
        if not VLM_AVAILABLE:
            raise RuntimeError("transformers/torch not installed or not importable.")
        if self.vlm_model is None or self.vlm_processor is None:
            print(f"[VLM] Loading {VLM_MODEL_ID} (this can take a bit the first time)...")

            # NOTE:
            # - This uses 4-bit quantization if bitsandbytes is available.
            # - If it fails, fallback to fp16 on GPU (needs more VRAM).
            load_kwargs = dict(
                device_map="auto",
                trust_remote_code=True,
                # Avoid FlashAttention2 if not installed
                attn_implementation="eager",
                use_flash_attention_2=False,
            )

            # Try 4-bit
            try:
                load_kwargs.update(dict(load_in_4bit=True))
            except Exception:
                pass

            self.vlm_processor = AutoProcessor.from_pretrained(VLM_MODEL_ID, trust_remote_code=True)
            cfg = AutoConfig.from_pretrained(VLM_MODEL_ID, trust_remote_code=True)
            # Explicitly disable FlashAttention2 to avoid hard failure if flash_attn isn't installed
            setattr(cfg, "attn_implementation", "eager")
            setattr(cfg, "use_flash_attention_2", False)
            self.vlm_model = AutoModelForCausalLM.from_pretrained(
                VLM_MODEL_ID, config=cfg, **load_kwargs
            )
            self.vlm_model.eval()
            try:
                print("[VLM] device_map:", getattr(self.vlm_model, "hf_device_map", None))
                print("[VLM] first param device:", next(self.vlm_model.parameters()).device)
                print("[VLM] cuda available:", torch.cuda.is_available())
            except Exception as e:
                print(f"[VLM] device check failed: {e}")
            print("[VLM] Ready.")

    @staticmethod
    def _extract_first_json(text: str) -> Optional[dict]:
        # find the first {...} block
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return None
        block = m.group(0)
        try:
            return json.loads(block)
        except Exception:
            return None

    def _run_vlm_weight_blocking(self, label: str, crop_bgr: np.ndarray) -> dict:
        """
        Blocking VLM call. Returns JSON-like dict with weight estimate.
        """
        self._ensure_vlm()

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)

        prompt = (
            "You are a careful assistant estimating OBJECT WEIGHT from a single image crop.\n"
            "Return ONLY a JSON object with this schema:\n"
            "{"
            '  "label": string,'
            '  "weight_g": {"min": int, "max": int},'
            '  "confidence": float,'
            '  "assumptions": [string]'
            "}\n"
            "Rules:\n"
            "- weight must be a plausible RANGE in grams.\n"
            "- if uncertain, widen the range and lower confidence.\n"
            f'Object label hint: "{label}".\n'
        )

        # Processor/model invocation varies across vision models; this is a robust-ish pattern:
        # Some VLMs require explicit image tags in the text prompt.
        if hasattr(self.vlm_processor, "apply_chat_template") and hasattr(
            self.vlm_processor, "chat_template"
        ):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"},
                    ],
                }
            ]
            prompt_text = self.vlm_processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        else:
            # Fallback: add a generic image placeholder token
            prompt_text = f"<|image_1|>\n{prompt}"

        inputs = self.vlm_processor(text=prompt_text, images=crop_pil, return_tensors="pt")
        # Move to model device
        for k in list(inputs.keys()):
            try:
                inputs[k] = inputs[k].to(self.vlm_model.device)
            except Exception:
                pass

        with torch.no_grad():
            out = self.vlm_model.generate(**inputs, max_new_tokens=VLM_MAX_NEW_TOKENS)

        # Robust decode: convert token IDs to python ints on CPU to avoid overflow
        try:
            ids = out[0].detach().cpu().tolist()
            # Filter/clamp to valid token id range to avoid overflow in fast tokenizers
            vocab_size = getattr(self.vlm_processor.tokenizer, "vocab_size", None)
            if vocab_size:
                ids = [int(i) for i in ids if 0 <= int(i) < vocab_size]
            else:
                ids = [int(i) for i in ids]
            text = self.vlm_processor.tokenizer.decode(ids, skip_special_tokens=True)
        except Exception:
            decoded = self.vlm_processor.batch_decode(out, skip_special_tokens=True)
            text = decoded[0] if decoded else ""
        js = self._extract_first_json(text)
        if js is None:
            # fallback: return something still usable
            return {
                "label": label,
                "weight_g": {"min": 100, "max": 3000},
                "confidence": 0.2,
                "assumptions": ["VLM output was not valid JSON", "fallback range"],
                "raw": text[:400],
            }
        # ensure label
        js.setdefault("label", label)
        return js

    async def _vlm_loop(self):
        """
        Consumes queued (label, crop) jobs. Only queued once per new label.
        """
        while True:
            try:
                label, crop = await self._vlm_queue.get()
                self._weight_by_label[label] = {"status": "running"}
                res = await asyncio.to_thread(self._run_vlm_weight_blocking, label, crop)
                self._weight_by_label[label] = res
                print(f"[VLM] result for {label}: {res}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[VLM] error: {e}")
                print(traceback.format_exc())

    # ======================
    # Live stream (Tobii)
    # ======================
    def start_live_stream(self, g3: Glasses3) -> None:
        self._init_ros()

        async def live_stream():
            async with g3.stream_rtsp(scene_camera=True, gaze=True) as streams:
                async with streams.scene_camera.decode() as scene_stream, streams.gaze.decode() as gaze_stream:
                    live_screen = self.get_screen("control").ids.sm.get_screen("live")
                    Window.bind(on_resize=live_screen.clear)

                    self.latest_frame_with_timestamp = await scene_stream.get()
                    self.latest_gaze_with_timestamp = await gaze_stream.get()

                    # Start background tasks
                    if self._detector_task is None or self._detector_task.done():
                        self._detector_task = self.create_task(self._detector_loop(DETECTOR_RATE_HZ), name="detector_loop")

                    if self._vlm_task is None or self._vlm_task.done():
                        self._vlm_task = self.create_task(self._vlm_loop(), name="vlm_loop")

                    self.read_frames_task = self.create_task(
                        update_frame(scene_stream, gaze_stream, streams),
                        name="update_frame",
                    )

                    if self.live_gaze_circle is None:
                        display = live_screen.ids.display
                        video_height = display.size[0] * VIDEO_Y_TO_X_RATIO
                        video_origin_y = (display.size[1] - video_height) / 2
                        self.live_gaze_circle = GazeCircle(
                            live_screen.ids.display.canvas,
                            (0, video_origin_y),
                            (display.size[0], video_height),
                        )

                        def _update_gaze_layout(*_):
                            video_h = display.width * VIDEO_Y_TO_X_RATIO
                            video_y = (display.height - video_h) / 2.0
                            if self.live_gaze_circle:
                                self.live_gaze_circle.origin = (0, video_y)
                                self.live_gaze_circle.size = (display.width, video_h)

                        _update_gaze_layout()
                        display.bind(size=_update_gaze_layout, pos=_update_gaze_layout)

                    self.draw_frame_event = Clock.schedule_interval(draw_frame, 1 / LIVE_FRAME_RATE)
                    await self.read_frames_task

        async def update_frame(scene_stream, gaze_stream, streams):
            while True:
                latest_frame_with_timestamp = await scene_stream.get()
                latest_gaze_with_timestamp = await gaze_stream.get()
                while (latest_gaze_with_timestamp[1] is None or latest_frame_with_timestamp[1] is None):
                    if latest_frame_with_timestamp[1] is None:
                        latest_frame_with_timestamp = await scene_stream.get()
                    if latest_gaze_with_timestamp[1] is None:
                        latest_gaze_with_timestamp = await gaze_stream.get()

                while latest_gaze_with_timestamp[1] < latest_frame_with_timestamp[1]:
                    latest_gaze_with_timestamp = await gaze_stream.get()
                    while latest_gaze_with_timestamp[1] is None:
                        latest_gaze_with_timestamp = await gaze_stream.get()

                self.latest_frame_with_timestamp = latest_frame_with_timestamp
                self.latest_gaze_with_timestamp = latest_gaze_with_timestamp
                logging.debug(streams.scene_camera.stats)

        def draw_frame(dt):
            if (self.latest_frame_with_timestamp is None or self.latest_gaze_with_timestamp is None or self.live_gaze_circle is None):
                logging.warning("Frame not drawn due to missing frame, gaze data or gaze circle.")
                return

            display = self.get_screen("control").ids.sm.get_screen("live").ids.display
            image = np.flip(self.latest_frame_with_timestamp[0].to_ndarray(format="bgr24"), 0)

            if not image.flags["C_CONTIGUOUS"]:
                image = np.ascontiguousarray(image)

            h, w = image.shape[:2]
            flip_label_vert = True

            if SHOW_OVERLAY:
                # draw detections
                for d in getattr(self, "_detections_latest", []):
                    try:
                        bb = self._clamp_xyxy(d["xyxy"], w, h)
                        if bb is None:
                            continue
                        x1, y1, x2, y2 = bb

                        # bbox
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # label text: include grasp state + (optional) weight if available
                        label = str(d.get("label", "")).strip()
                        label_state = str(d.get("label_state", label)).strip()

                        weight_txt = ""
                        if label in self._weight_by_label and isinstance(self._weight_by_label[label], dict):
                            wd = self._weight_by_label[label]
                            if "weight_g" in wd and isinstance(wd["weight_g"], dict):
                                mn = wd["weight_g"].get("min", None)
                                mx = wd["weight_g"].get("max", None)
                                if mn is not None and mx is not None:
                                    weight_txt = f" | {mn}-{mx}g"
                            elif wd.get("status") == "queued":
                                weight_txt = " | weight: queued"
                            elif wd.get("status") == "running":
                                weight_txt = " | weight: running"

                        label_text = f"{label_state}{weight_txt}"

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.55
                        thickness = 2
                        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                        pad = 4

                        patch_w = tw + 2 * pad
                        patch_h = th + baseline + 2 * pad

                        top_y = y1 - 4 - patch_h
                        left_x = x1
                        if top_y < 0:
                            top_y = y1 + 2

                        patch = np.zeros((patch_h, patch_w, 3), dtype=np.uint8)
                        cv2.putText(
                            patch,
                            label_text,
                            (pad, patch_h - baseline - pad),
                            font,
                            font_scale,
                            (255, 255, 255),
                            thickness,
                            cv2.LINE_AA,
                        )

                        if flip_label_vert:
                            patch = cv2.flip(patch, 0)

                        x0 = max(0, left_x)
                        y0 = max(0, top_y)
                        x_end = min(w, left_x + patch_w)
                        y_end = min(h, top_y + patch_h)
                        if x_end > x0 and y_end > y0:
                            px0 = max(0, -left_x)
                            py0 = max(0, -top_y)
                            px_end = px0 + (x_end - x0)
                            py_end = py0 + (y_end - y0)
                            image[y0:y_end, x0:x_end] = patch[py0:py_end, px0:px_end]

                    except Exception as e:
                        print(f"[DRAW] label error: {e}")

            texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt="bgr")
            flat = np.reshape(image, -1)
            texture.blit_buffer(flat, colorfmt="bgr", bufferfmt="ubyte")

            display.canvas.add(Color(1, 1, 1, 1))
            if self.last_texture is not None:
                display.canvas.remove(self.last_texture)

            video_h = display.width * VIDEO_Y_TO_X_RATIO
            video_y = (display.height - video_h) / 2.0
            self.last_texture = Rectangle(texture=texture, pos=(0, video_y), size=(display.width, video_h))
            display.canvas.add(self.last_texture)

            gaze_data = self.latest_gaze_with_timestamp[0]
            if len(gaze_data) != 0:
                point = gaze_data["gaze2d"]
                self.live_gaze_circle.redraw(point)
                self._print_gaze(self.latest_gaze_with_timestamp[1], gaze_data)

        def live_stream_task_running() -> bool:
            return (self.live_stream_task is not None) and (not self.live_stream_task.done())

        if live_stream_task_running():
            logging.info("Task not started: live_stream_task already running.")
        else:
            self.live_stream_task = self.create_task(live_stream(), name="live_stream_task")

    async def stop_live_stream(self) -> None:
        # cancel background tasks
        if self._detector_task is not None and not self._detector_task.cancelled():
            await self.cancel_task(self._detector_task)
        if self._vlm_task is not None and not self._vlm_task.cancelled():
            await self.cancel_task(self._vlm_task)

        if self.read_frames_task is not None and not self.read_frames_task.cancelled():
            await self.cancel_task(self.read_frames_task)
        if self.live_stream_task is not None and not self.live_stream_task.cancelled():
            await self.cancel_task(self.live_stream_task)

        if self.draw_frame_event is not None:
            self.draw_frame_event.cancel()
            self.draw_frame_event = None

        live_screen = self.get_screen("control").ids.sm.get_screen("live")
        Window.unbind(on_resize=live_screen.clear)
        live_screen.clear()
        self.last_texture = None
        self._shutdown_ros()

    # ---------- Utils ----------
    def _print_gaze(self, gaze_ts, gaze_data, rate_hz: float = 5.0):
        now = time.time()
        if now - self._last_gaze_print < 1.0 / rate_hz:
            return
        self._last_gaze_print = now
        try:
            _ = gaze_data["gaze2d"]
        except Exception as e:
            print(f"[LIVE] gaze missing/invalid: {e}")

    # ======================
    # App wiring (same)
    # ======================
    def build(self):
        return self

    def on_start(self):
        # Eagerly load VLM at startup (non-blocking) to avoid first-use latency.
        self.create_task(self._warm_vlm(), name="vlm_warmup")
        self.create_task(self.backend_app(), name="backend_app")
        self.send_app_event(AppEventKind.START_DISCOVERY)

    async def _warm_vlm(self) -> None:
        try:
            await asyncio.to_thread(self._ensure_vlm)
        except Exception as e:
            print(f"[VLM] warmup failed: {e}")

    def close(self, *args) -> bool:
        self.send_app_event(AppEventKind.STOP)
        return True

    def switch_to_screen(self, screen: str):
        self.transition.direction = "right" if screen == "discovery" else "left"
        self.current = screen

    def start_control(self) -> bool:
        selected = self.get_screen("discovery").ids.services.ids.selectables.selected_nodes
        if len(selected) <= 0:
            popup = UserMessagePopup(title="No Glasses3 unit selected")
            popup.ids.message_label.text = "Please select a Glasses3 unit and try again."
            popup.open()
            return False
        else:
            # NOTE: your original code hard-coded the hostname. Keeping same behavior:
            hostname = "192.168.75.51"
            self.backend_control_task = self.create_task(self.backend_control(hostname), name="backend_control")
            self.get_screen("control").set_hostname(hostname)
            self.switch_to_screen("control")
            return True

    async def stop_control(self) -> None:
        await self.cancel_task(self.backend_control_task)
        self.get_screen("control").clear()

    def start_discovery(self):
        self.discovery_task = self.create_task(self.backend_discovery(), name="backend_discovery")
        self.switch_to_screen("discovery")

    async def stop_discovery(self):
        await self.cancel_task(self.discovery_task)
        self.get_screen("discovery").clear()

    def send_app_event(self, event: AppEventKind) -> None:
        self.app_events.put_nowait(event)

    async def backend_app(self) -> None:
        while True:
            app_event = await self.app_events.get()
            await self.handle_app_event(app_event)
            if app_event == AppEventKind.STOP:
                break

    async def handle_app_event(self, event: AppEventKind):
        logging.info(f"Handling app event: {event}")
        match event:
            case AppEventKind.START_DISCOVERY:
                self.start_discovery()
            case AppEventKind.ENTER_CONTROL_SESSION:
                if self.start_control():
                    await self.stop_discovery()
            case AppEventKind.LEAVE_CONTROL_SESSION:
                self.start_discovery()
                await self.stop_control()
            case AppEventKind.STOP:
                match self.current:
                    case "discovery":
                        await self.stop_discovery()
                    case "control":
                        await self.stop_control()
                self.stop()

    async def backend_discovery(self) -> None:
        async with G3ServiceDiscovery.listen() as service_listener:
            while True:
                await self.handle_service_event(await service_listener.events.get())

    async def handle_service_event(self, event: Tuple[EventKind, G3Service]) -> None:
        logging.info(f"Handling service event: {event[0]}")
        match event:
            case (EventKind.ADDED, service):
                self.get_screen("discovery").add_service(service.hostname, service.ipv4_address, service.ipv6_address)
            case (EventKind.UPDATED, service):
                self.get_screen("discovery").update_service(service.hostname, service.ipv4_address, service.ipv6_address)
            case (EventKind.REMOVED, service):
                self.get_screen("discovery").remove_service(service.hostname, service.ipv4_address, service.ipv6_address)

    def send_control_event(self, event: ControlEventKind) -> None:
        self.control_events.put_nowait(event)

    async def backend_control(self, hostname: str) -> None:
        async with connect_to_glasses.with_hostname(hostname) as g3:
            async with g3.recordings.keep_updated_in_context():
                update_recordings_task = self.create_task(
                    self.update_recordings(g3, g3.recordings.events),
                    name="update_recordings",
                )
                await self.start_update_recorder_status(g3)
                try:
                    while True:
                        await self.handle_control_event(g3, await self.control_events.get())
                finally:
                    await self.cancel_task(update_recordings_task)
                    await self.stop_update_recorder_status()

    async def handle_control_event(self, g3: Glasses3, event: ControlEventKind) -> None:
        logging.info(f"Handling control event: {event}")
        self.get_screen("control").set_task_running_status(True)
        match event:
            case ControlEventKind.START_RECORDING:
                await g3.recorder.start()
            case ControlEventKind.STOP_RECORDING:
                await g3.recorder.stop()
            case ControlEventKind.DELETE_RECORDING:
                await self.delete_selected_recording(g3)
            case ControlEventKind.START_LIVE:
                self.start_live_stream(g3)
            case ControlEventKind.STOP_LIVE:
                await self.stop_live_stream()
            case ControlEventKind.PLAY_RECORDING:
                await self.play_selected_recording(g3)
        self.get_screen("control").set_task_running_status(False)

    async def delete_selected_recording(self, g3: Glasses3) -> None:
        uuid = self.get_selected_recording()
        if uuid is not None:
            await g3.recordings.delete(uuid)

    def get_selected_recording(self) -> Optional[str]:
        recordings = self.get_screen("control").ids.sm.get_screen("recorder").ids.recordings
        selected = recordings.ids.selectables.selected_nodes
        if len(selected) != 1:
            popup = UserMessagePopup(title="No recording selected")
            popup.ids.message_label.text = "Please select a recording and try again."
            popup.open()
        else:
            return recordings.data[selected[0]]["uuid"]

    async def play_selected_recording(self, g3: Glasses3) -> None:
        uuid = self.get_selected_recording()
        if uuid is not None:
            self.get_screen("control").switch_to_screen("recording")
            recording = g3.recordings.get_recording(uuid)
            file_url = await recording.get_scenevideo_url()
            videoplayer = self.get_screen("control").ids.sm.get_screen("recording").ids.videoplayer
            videoplayer.source = file_url
            videoplayer.state = "play"

            async with aiohttp.ClientSession() as session:
                async with session.get(await recording.get_gazedata_url()) as response:
                    all_gaze_data = await response.text()
            gaze_json_list = all_gaze_data.split("\n")[:-1]
            self.gaze_data_list = [json.loads(g) for g in gaze_json_list]

            if self.replay_gaze_circle is None:
                video_height = videoplayer.size[0] * VIDEO_Y_TO_X_RATIO
                video_origin_y = (videoplayer.size[1] - video_height + VIDEOPLAYER_PROGRESS_BAR_HEIGHT) / 2
                self.replay_gaze_circle = GazeCircle(
                    videoplayer.canvas,
                    (0, video_origin_y),
                    (videoplayer.size[0], video_height),
                )
                self.bind_replay_gaze_updates()

    def bind_replay_gaze_updates(self):
        def reset_gaze_circle(instance, state):
            if state in ("start", "stop"):
                if self.replay_gaze_circle is not None:
                    self.replay_gaze_circle.reset()

        def update_gaze_circle(instance, timestamp):
            if self.replay_gaze_circle is None:
                logging.warning("Gaze not drawn due to missing gaze circle.")
                return
            current_gaze_index = self.binary_search_gaze_point(timestamp, self.gaze_data_list)
            try:
                point = self.gaze_data_list[current_gaze_index]["data"]["gaze2d"]
            except KeyError:
                point = None
            self.replay_gaze_circle.redraw(point)

        videoplayer = self.get_screen("control").ids.sm.get_screen("recording").ids.videoplayer
        videoplayer.bind(position=update_gaze_circle)
        videoplayer.bind(state=reset_gaze_circle)

    @staticmethod
    def binary_search_gaze_point(value, gaze_list):
        left_index = 0
        right_index = len(gaze_list) - 1
        best_index = left_index
        while left_index <= right_index:
            mid_index = left_index + (right_index - left_index) // 2
            if gaze_list[mid_index]["timestamp"] < value:
                left_index = mid_index + 1
            elif gaze_list[mid_index]["timestamp"] > value:
                right_index = mid_index - 1
            else:
                best_index = mid_index
                break
            if abs(gaze_list[mid_index]["timestamp"] - value) < abs(gaze_list[best_index]["timestamp"] - value):
                best_index = mid_index
        return best_index

    async def update_recordings(self, g3, recordings_events):
        recorder_screen = self.get_screen("control").ids.sm.get_screen("recorder")
        for child in cast(List[Recording], g3.recordings):
            recorder_screen.add_recording(await child.get_visible_name(), child.uuid, child, atEnd=True)
        while True:
            event = await recordings_events.get()
            match event:
                case (RecordingsEventKind.ADDED, body):
                    uuid = cast(List[str], body)[0]
                    recording = g3.recordings.get_recording(uuid)
                    recorder_screen.add_recording(await recording.get_visible_name(), recording.uuid, recording)
                case (RecordingsEventKind.REMOVED, body):
                    uuid = cast(List[str], body)[0]
                    recorder_screen.remove_recording(uuid)

    async def start_update_recorder_status(self, g3: Glasses3) -> None:
        recorder_screen = self.get_screen("control").ids.sm.get_screen("recorder")
        recorder_screen.set_recording_status(True if await g3.recorder.get_created() is not None else False)
        (recorder_started_queue, self.unsubscribe_to_recorder_started) = await g3.recorder.subscribe_to_started()
        (recorder_stopped_queue, self.unsubscribe_to_recorder_stopped) = await g3.recorder.subscribe_to_stopped()

        async def handle_recorder_started(q: asyncio.Queue[SignalBody]):
            while True:
                await q.get()
                recorder_screen.set_recording_status(True)

        async def handle_recorder_stopped(q: asyncio.Queue[SignalBody]):
            while True:
                await q.get()
                recorder_screen.set_recording_status(False)

        self.handle_recorder_started_task = self.create_task(
            handle_recorder_started(recorder_started_queue), name="handle_recorder_started"
        )
        self.handle_recorder_stopped_task = self.create_task(
            handle_recorder_stopped(recorder_stopped_queue), name="handle_recorder_stopped"
        )

    async def stop_update_recorder_status(self) -> None:
        await self.unsubscribe_to_recorder_started
        await self.unsubscribe_to_recorder_stopped
        await self.cancel_task(self.handle_recorder_started_task)
        await self.cancel_task(self.handle_recorder_stopped_task)

    def create_task(self, coro, name=None) -> asyncio.Task:
        task = asyncio.create_task(coro, name=name)
        logging.info(f"Task created: {task.get_name()}")
        self.tasks.add(task)
        task.add_done_callback(self.tasks.remove)
        return task

    async def cancel_task(self, task: asyncio.Task) -> None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            logging.info(f"Task cancelled: {task.get_name()}")


if __name__ == "__main__":
    app = G3App()
    asyncio.run(app.async_run())
