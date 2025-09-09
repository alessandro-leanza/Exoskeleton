#!/usr/bin/env python3
import asyncio
import json
import logging
from typing import List, Optional, Set, Tuple, cast

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

# YOLO + OpenCV
from ultralytics import YOLO
import torch
import cv2
from collections import deque, Counter
import time

# ROS2
import rclpy
from rclpy.node import Node
from exo_interfaces.srv import RunTrajectory, SetAdmittanceParams
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState

# ====================== PARAMS ======================

# Gaze dwell
GAZE_DWELL_WINDOW = 15
GAZE_DWELL_RATIO  = 0.6

# Service names & admittance
RUN_TRAJ_SRV_NAME = 'run_trajectory'
SET_ADMITT_SRV_NAME = 'set_admittance_params'
ACTION_COOLDOWN_S = 1.0
K_DOWN = 30.0
K_UP   = 30.0

# History (se mai volessi tornare a ratio su classi)
HISTORY_LENGTH = 20
WARMUP = 20
TRIGGER_PERCENTAGE_NOT_GRASPED = 0.7
TRIGGER_PERCENTAGE_GRASPED = 0.4

YOLO_MODEL_PATH = '/home/alessandro/exo_v2_ws/src/exo_control/yolo_weights/best-realsense-tobii.pt'

GAZE_CIRCLE_RADIUS = 10
VIDEOPLAYER_PROGRESS_BAR_HEIGHT = dp(44)
VIDEO_Y_TO_X_RATIO = 9 / 16
LIVE_FRAME_RATE = 25

# Dwell asimmetrico su gaze→bbox
GAZE_DWELL_WIN_DOWN    = 10
GAZE_DWELL_RATIO_DOWN  = 0.60
GAZE_DWELL_WIN_UP      = 8
GAZE_DWELL_RATIO_UP    = 0.60
N_CONSEC_GAZE_DOWN = 0
N_CONSEC_GAZE_UP   = 5

# Freeze tra azioni e warmup
POST_ACTION_FREEZE_S = 1.5
POST_ACTION_MIN_SAMPLES = 3

# Place via soglie relative JointState
DELTA_DOWN_PLACE  = 0.04
DELTA_UP_PLACE    = 0.05
ARM_DELAY_S       = 2.5     # baseline dopo 2.5 s

JOINT_NAME     = 'joint_0'
FALLBACK_INDEX = 0

logging.basicConfig(level=logging.DEBUG)

# ====================== KIVY KV ======================
Builder.load_string("""
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
""")

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
        self.tasks: Set[asyncio.Task] = set()
        self.app_events: asyncio.Queue[AppEventKind] = asyncio.Queue()
        self.control_events: asyncio.Queue[ControlEventKind] = asyncio.Queue()
        self.live_stream_task: Optional[asyncio.Task] = None
        self.read_frames_task: Optional[asyncio.Task] = None
        self.add_widget(DiscoveryScreen(name="discovery"))
        self.add_widget(ControlScreen(name="control"))
        self.latest_frame_with_timestamp = None
        self.latest_gaze_with_timestamp = None
        self.live_gaze_circle = None
        self.replay_gaze_circle = None
        self.last_texture = None
        self.draw_frame_event = None

        self._last_gaze_print = 0.0

        # YOLO
        self.yolo_model = None
        self._yolo_task = None
        self._yolo_latest = []

        # FSM pick
        self.gaze_on_box_history = deque(maxlen=max(GAZE_DWELL_WIN_DOWN, GAZE_DWELL_WIN_UP))
        self.state = 'stand_no_box'      # 'bend_no_box', 'stand_with_box', 'bend_to_place'
        self.last_action = None
        self._last_action_time = 0.0
        self._block_until = 0.0
        self._post_unblock_samples = 0

        # ROS
        self.ros_node: Node | None = None
        self.traj_client = None
        self.adm_client = None
        self.box_gate_pub = None
        self.js_sub = None
        self._ros_ready = False
        self.box_gate = False

        # Place baselines
        self.jidx = None
        self.last_pos = None
        self.baseline_swB = None
        self.baseline_b2p = None
        self.arm_until = 0.0  # quando < now, i trigger place sono armati

    # ---------- GAZE in bbox ----------
    def _gaze_in_any_box(self, det_list, frame_w: int, frame_h: int) -> bool:
        """
        True se lo sguardo cade dentro QUALSIASI bbox YOLO relativo al box,
        senza distinguere tra 'Box-Grasped' e 'Box-Not Grasped'.
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

        gx = min(max(gx, 0.0), 1.0)
        gy = min(max(gy, 0.0), 1.0)
        x_px = gx * frame_w
        y_px = (1.0 - gy) * frame_h   # flip verticale (l’immagine è flippata)

        for d in det_list:
            label = d.get("label", "")
            # a prescindere dal “tipo” di box: basta sia una delle due classi box
            if label not in ("Box-Grasped", "Box-Not Grasped"):
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
        self.ros_node = rclpy.create_node('vision_command_node')
        self.traj_client = self.ros_node.create_client(RunTrajectory, RUN_TRAJ_SRV_NAME)
        self.adm_client  = self.ros_node.create_client(SetAdmittanceParams, SET_ADMITT_SRV_NAME)
        self.box_gate_pub = self.ros_node.create_publisher(Bool, 'perception/box_gate', 10)
        self.js_sub       = self.ros_node.create_subscription(JointState, 'joint_states', self._on_js, 30)

        # pub stato iniziale
        self.box_gate_pub.publish(Bool(data=False))
        self.box_gate = False

        if not self.traj_client.wait_for_service(timeout_sec=5.0):
            print(f"[ROS] ERROR: service '{RUN_TRAJ_SRV_NAME}' not available")
        if not self.adm_client.wait_for_service(timeout_sec=5.0):
            print(f"[ROS] ERROR: service '{SET_ADMITT_SRV_NAME}' not available")

        self._ros_ready = True
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
            self.traj_client = None
            self.adm_client = None
            self.box_gate_pub = None
            self.js_sub = None

    def _send_trajectory_request_blocking(self, direction: str):
        if not self._ros_ready or self.ros_node is None or self.traj_client is None:
            print("[ROS] traj client not ready")
            return
        req = RunTrajectory.Request()
        req.trajectory_type = direction
        future = self.traj_client.call_async(req)
        rclpy.spin_until_future_complete(self.ros_node, future)

    def _set_admittance_params_blocking(self, k_val: float):
        if not self._ros_ready or self.ros_node is None or self.adm_client is None:
            print("[ROS] adm client not ready")
            return
        req = SetAdmittanceParams.Request()
        req.k = k_val
        future = self.adm_client.call_async(req)
        rclpy.spin_until_future_complete(self.ros_node, future)

    async def _trigger_action_ros(self, action: str):
        if not self._ros_ready:
            print("[ROS] not ready, skipping action")
            return
        k_val = K_DOWN if action == 'down' else K_UP
        await asyncio.to_thread(self._set_admittance_params_blocking, k_val)
        await asyncio.to_thread(self._send_trajectory_request_blocking, action)

    def _publish_box_gate(self, val: bool):
        if not self._ros_ready:
            return
        if val != self.box_gate:
            self.box_gate = val
            self.box_gate_pub.publish(Bool(data=val))
            print(f"[BOX_GATE] → {val}")

    # ---------- Pick FSM basata SOLO su gaze→bbox ----------
    def _evaluate_gaze_and_trigger(self):
        now = time.monotonic()

        if now < self._block_until:
            return
        if self._post_unblock_samples < POST_ACTION_MIN_SAMPLES:
            return

        hist = list(self.gaze_on_box_history)
        n = len(hist)
        if n == 0:
            return

        action = None

        # Fast path
        if self.state == 'stand_no_box' and N_CONSEC_GAZE_DOWN > 0 and n >= N_CONSEC_GAZE_DOWN:
            if all(x == 1 for x in hist[-N_CONSEC_GAZE_DOWN:]):
                action = 'down'
                self.state = 'bend_no_box'
        elif self.state == 'bend_no_box' and N_CONSEC_GAZE_UP > 0 and n >= N_CONSEC_GAZE_UP:
            if all(x == 1 for x in hist[-N_CONSEC_GAZE_UP:]):
                action = 'up'
                self.state = 'stand_with_box'
                # Gate ON e preparo baselines + arming delay per il place
                self._publish_box_gate(True)
                self.baseline_swB = None
                self.baseline_b2p = None
                self.arm_until = time.monotonic() + ARM_DELAY_S
                print(f"[STATE] bend_no_box -> stand_with_box | arm {ARM_DELAY_S:.2f}s (fast)")

        # Ratio path
        if action is None:
            if self.state == 'stand_no_box':
                k = min(GAZE_DWELL_WIN_DOWN, n)
                ratio = sum(hist[-k:]) / k if k > 0 else 0.0
                if ratio >= GAZE_DWELL_RATIO_DOWN:
                    action = 'down'
                    self.state = 'bend_no_box'
            elif self.state == 'bend_no_box':
                k = min(GAZE_DWELL_WIN_UP, n)
                ratio = sum(hist[-k:]) / k if k > 0 else 0.0
                if ratio >= GAZE_DWELL_RATIO_UP:
                    action = 'up'
                    self.state = 'stand_with_box'
                    self._publish_box_gate(True)
                    self.baseline_swB = None
                    self.baseline_b2p = None
                    self.arm_until = time.monotonic() + ARM_DELAY_S
                    print(f"[STATE] bend_no_box -> stand_with_box | arm {ARM_DELAY_S:.2f}s (ratio)")

        if action is not None:
            if action != self.last_action or (now - self._last_action_time) >= ACTION_COOLDOWN_S:
                print(f"[ACTION] GAZE: {action.upper()}  | state -> {self.state}")
                self.last_action = action
                self._last_action_time = now
                self.create_task(self._trigger_action_ros(action), name=f"ros_action_{action}_{int(now)}")

                # Freeze + reset finestra
                self._block_until = now + POST_ACTION_FREEZE_S
                self._post_unblock_samples = 0
                self.gaze_on_box_history.clear()

    # ---------- JointState → place ----------
    def _on_js(self, msg: JointState):
        # risolvo indice al primo giro
        if self.jidx is None:
            if msg.name and JOINT_NAME in msg.name:
                self.jidx = msg.name.index(JOINT_NAME)
            else:
                self.jidx = min(FALLBACK_INDEX, len(msg.position) - 1)
            print(f"[MEAS READY] Using joint index {self.jidx} "
                  f"({'name:'+JOINT_NAME if msg.name else 'no names'})")
            return
        if self.jidx >= len(msg.position):
            return

        pos = float(msg.position[self.jidx])
        self.last_pos = pos
        now = time.monotonic()

        # stand_with_box → prendo baseline_swB dopo ARM_DELAY_S
        if self.state == 'stand_with_box':
            if self.baseline_swB is None:
                if now < self.arm_until:
                    print(f"[BASE WAIT] stand_with_box: waiting {self.arm_until - now:.2f}s")
                    return
                self.baseline_swB = pos
                print(f"[BASE] stand_with_box baseline_swB={pos:.3f} rad")
                return

            delta = pos - self.baseline_swB
            print(f"[Δ] stand_with_box: pos={pos:.3f} base={self.baseline_swB:.3f} Δ={delta:.3f} (thr={DELTA_DOWN_PLACE:.3f})")
            if delta >= DELTA_DOWN_PLACE:
                print(f"[TRIGGER] stand_with_box → DOWN (place)")
                self.state = 'bend_to_place'
                self.baseline_b2p = None
                self.arm_until = now + ARM_DELAY_S
                self.create_task(self._trigger_action_ros('down'), name=f"ros_action_down_{int(now)}")
                return

        # bend_to_place → baseline_b2p dopo ARM_DELAY_S
        elif self.state == 'bend_to_place':
            if self.baseline_b2p is None:
                if now < self.arm_until:
                    print(f"[BASE WAIT] bend_to_place: waiting {self.arm_until - now:.2f}s")
                    return
                self.baseline_b2p = pos
                print(f"[BASE] bend_to_place baseline_b2p={pos:.3f} rad")
                return

            delta = self.baseline_b2p - pos
            print(f"[Δ] bend_to_place: pos={pos:.3f} base={self.baseline_b2p:.3f} Δ={delta:.3f} (thr={DELTA_UP_PLACE:.3f})")
            if delta >= DELTA_UP_PLACE:
                print(f"[TRIGGER] bend_to_place → UP (return)")
                self.state = 'stand_no_box'
                self._publish_box_gate(False)
                self.baseline_swB = None
                self.baseline_b2p = None
                self.arm_until    = now + ARM_DELAY_S
                self.create_task(self._trigger_action_ros('up'), name=f"ros_action_up_{int(now)}")
                return

    # ---------- YOLO loop ----------
    async def _yolo_loop(self, conf: float = 0.7, rate_hz: float = 3.0):
        period = 1.0 / rate_hz
        while True:
            start = time.monotonic()
            try:
                fw = self.latest_frame_with_timestamp
                if not fw or fw[0] is None:
                    await asyncio.sleep(0.05)
                    continue

                # frame BGR flippato verticalmente (coerente con draw)
                frame = np.flip(fw[0].to_ndarray(format="bgr24"), 0)

                # inferenza in thread separato
                results = await asyncio.to_thread(self.yolo_model.predict, frame, conf=conf, verbose=False)
                r0 = results[0]
                boxes = r0.boxes

                det_list = []
                if boxes is None or len(boxes) == 0:
                    print("[YOLO] No detections")
                else:
                    out = []
                    for b in boxes:
                        cls_id = int(b.cls[0])
                        p = float(b.conf[0])
                        label = self.yolo_model.names.get(cls_id, str(cls_id))
                        out.append(f"{label}({p:.2f})")
                        xyxy = b.xyxy[0].detach().cpu().numpy().astype(int).tolist()
                        det_list.append({"xyxy": xyxy, "label": label, "conf": p})
                    print("[YOLO] Detected:", ", ".join(out))

                self._yolo_latest = det_list

                # === GAZE DWELL: aggiorna history basata SOLO su gaze in bbox ===
                h, w = frame.shape[:2]
                gaze_on_box = self._gaze_in_any_box(det_list, w, h)

                now_loop = time.monotonic()
                if now_loop >= self._block_until:
                    self.gaze_on_box_history.append(1 if gaze_on_box else 0)
                    self._post_unblock_samples += 1
                    self._evaluate_gaze_and_trigger()
                # se in freeze, ignoro

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[YOLO] Error during inference: {e}")

            # processa JointState senza bloccare
            if self._ros_ready:
                try:
                    rclpy.spin_once(self.ros_node, timeout_sec=0.0)
                except Exception:
                    pass

            dt = time.monotonic() - start
            await asyncio.sleep(max(0.0, period - dt))

    # ---------- Live stream ----------
    def start_live_stream(self, g3: Glasses3) -> None:
        self._init_ros()

        async def live_stream():
            async with g3.stream_rtsp(scene_camera=True, gaze=True) as streams:
                async with streams.scene_camera.decode() as scene_stream, streams.gaze.decode() as gaze_stream:
                    if self.yolo_model is None:
                        print(f"[YOLO] Loading model: {YOLO_MODEL_PATH}")
                        self.yolo_model = YOLO(YOLO_MODEL_PATH)
                        print("[YOLO] Using CUDA" if torch.cuda.is_available() else "[YOLO] Using CPU")

                    if self._yolo_task is None or self._yolo_task.done():
                        self._yolo_task = self.create_task(self._yolo_loop(conf=0.7, rate_hz=3.0), name="yolo_loop")

                    live_screen = self.get_screen("control").ids.sm.get_screen("live")
                    Window.bind(on_resize=live_screen.clear)
                    self.latest_frame_with_timestamp = await scene_stream.get()
                    self.latest_gaze_with_timestamp  = await gaze_stream.get()

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

                    self.draw_frame_event = Clock.schedule_interval(draw_frame, 1 / LIVE_FRAME_RATE)
                    await self.read_frames_task

        async def update_frame(scene_stream, gaze_stream, streams):
            while True:
                latest_frame_with_timestamp = await scene_stream.get()
                latest_gaze_with_timestamp  = await gaze_stream.get()
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
                self.latest_gaze_with_timestamp  = latest_gaze_with_timestamp
                logging.debug(streams.scene_camera.stats)

        def draw_frame(dt):
            if (self.latest_frame_with_timestamp is None
                or self.latest_gaze_with_timestamp is None
                or self.live_gaze_circle is None):
                logging.warning("Frame not drawn due to missing frame, gaze data or gaze circle.")
                return

            display = self.get_screen("control").ids.sm.get_screen("live").ids.display
            image = np.flip(self.latest_frame_with_timestamp[0].to_ndarray(format="bgr24"), 0)

            if not image.flags["C_CONTIGUOUS"]:
                image = np.ascontiguousarray(image)

            h, w = image.shape[:2]
            for d in getattr(self, "_yolo_latest", []):
                x1, y1, x2, y2 = d["xyxy"]
                x1 = max(0, min(w - 1, int(x1)))
                y1 = max(0, min(h - 1, int(y1)))
                x2 = max(0, min(w - 1, int(x2)))
                y2 = max(0, min(h - 1, int(y2)))
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    f'{d["label"]} {d["conf"]:.2f}',
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt="bgr")
            image = np.reshape(image, -1)
            texture.blit_buffer(image, colorfmt="bgr", bufferfmt="ubyte")
            display.canvas.add(Color(1, 1, 1, 1))
            if self.last_texture is not None:
                display.canvas.remove(self.last_texture)
            self.last_texture = Rectangle(
                texture=texture,
                pos=(0, (display.top - display.width * VIDEO_Y_TO_X_RATIO) / 2),
                size=(display.width, display.width * VIDEO_Y_TO_X_RATIO),
            )
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
            pt = gaze_data["gaze2d"]
            print(f"[LIVE] gaze2d=({pt[0]:.3f}, {pt[1]:.3f}) ts={gaze_ts:.3f}s")
        except Exception as e:
            print(f"[LIVE] gaze missing/invalid: {e}")

    # ---------- App wiring ----------
    def build(self):
        return self

    def on_start(self):
        self.create_task(self.backend_app(), name="backend_app")
        self.send_app_event(AppEventKind.START_DISCOVERY)

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
        recorder_screen.set_recording_status(True if await g3.recorder.get_created() != None else False)
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
            handle_recorder_started(recorder_started_queue), name="handle_recorder_started")
        self.handle_recorder_stopped_task = self.create_task(
            handle_recorder_stopped(recorder_stopped_queue), name="handle_recorder_stopped")

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
