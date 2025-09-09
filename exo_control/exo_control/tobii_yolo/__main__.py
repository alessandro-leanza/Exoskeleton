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

# MODIFICA: import YOLO model
from ultralytics import YOLO
import torch, time

# MODIFICA: opencv to draw bboxes
import cv2

# MODIFICA: librerie per azioni
from collections import deque, Counter
import time

import rclpy
from rclpy.node import Node
from exo_interfaces.srv import RunTrajectory, SetAdmittanceParams

from std_msgs.msg import Bool
from sensor_msgs.msg import JointState

# Service names & params
RUN_TRAJ_SRV_NAME = 'run_trajectory'
SET_ADMITT_SRV_NAME = 'set_admittance_params'
ACTION_COOLDOWN_S = 0.0
K_DOWN = 30.0
K_UP   = 30.0

HISTORY_LENGTH = 10          # ~3s a 30 FPS se inferenza ~10 Hz -> regola a piacere
WARMUP = 0                  # frame minimi prima di valutare trigger
TRIGGER_PERCENTAGE_NOT_GRASPED = 0.7
TRIGGER_PERCENTAGE_GRASPED = 0.6
ACTION_COOLDOWN_S = 1.0


YOLO_MODEL_PATH = '/home/alessandro/exo_v2_ws/src/exo_control/yolo_weights/best-realsense-tobii.pt'

GAZE_CIRCLE_RADIUS = 10
VIDEOPLAYER_PROGRESS_BAR_HEIGHT = dp(44)
VIDEO_Y_TO_X_RATIO = 9 / 16
LIVE_FRAME_RATE = 25

# Finestre e soglie asimmetriche
TAIL_NOT_GRASPED   = 12   # quanti frame considerare per scendere
TAIL_GRASPED       = 6    # quanti frame considerare per salire (più corto => più reattivo)
THRESH_NOT_GRASPED = 0.6 # % di "not_grasped" per scendere
THRESH_GRASPED     = 0.6 # % di "grasped" per salire

# Fast-path: trigger immediato con N consecutivi (opzionale)
N_CONSEC_NOT_GRASPED = 0  # 0 = disattivo; oppure 3-4
N_CONSEC_GRASPED     = 3  # pochi consecutivi per far salire più veloce

# --- Place via soglie relative sul JointState ---
DELTA_DOWN_PLACE  = 0.04   # stand_with_box -> (DOWN) -> bend_to_place: pos - baseline_swB >= soglia
DELTA_UP_PLACE    = 0.05   # bend_to_place   -> (UP)   -> stand_without_box: baseline_b2p - pos >= soglia
ARM_DELAY_S       = 3.2   # ritardo anti-transitorio prima di fissare la baseline

# Joint da monitorare (se msg.name è presente):
JOINT_NAME     = 'joint_0'
FALLBACK_INDEX = 0


logging.basicConfig(level=logging.DEBUG)

# fmt: off
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
"""
)
# fmt: on


class SelectableRecycleBoxLayout(LayoutSelectionBehavior, RecycleBoxLayout):
    pass


class SelectableLabel(RecycleDataViewBehavior, Label):
    index = None
    selected = BooleanProperty(False)
    selectable = BooleanProperty(True)

    def refresh_view_attrs(self, rv, index, data):
        """Catch and handle the view changes"""
        self.index = index
        return super().refresh_view_attrs(rv, index, data)

    def on_touch_down(self, touch):
        """Add selection on touch down"""
        if super().on_touch_down(touch):
            return True
        if self.collide_point(*touch.pos) and self.selectable:
            return self.parent.select_with_touch(self.index, touch)

    def apply_selection(self, rv, index, is_selected):
        """Respond to the selection of items in the view."""
        self.selected = is_selected


class SelectableList(RecycleView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = []


class DiscoveryScreen(Screen):
    def add_service(
        self, hostname: str, ipv4: Optional[str], ipv6: Optional[str]
    ) -> None:
        self.ids.services.data.append(
            {"hostname": hostname, "text": f"{hostname}\n{ipv4}\n{ipv6}"}
        )
        logging.info(f"Services: Added {hostname}, {ipv4}, {ipv6}")

    def update_service(
        self, hostname: str, ipv4: Optional[str], ipv6: Optional[str]
    ) -> None:
        services = self.ids.services
        for service in services.data:
            if service["hostname"] == hostname:
                service["text"] = f"{hostname}\n{ipv4}\n{ipv6}"
                logging.info(f"Services: Updated {hostname}, {ipv4}, {ipv6}")

    def remove_service(
        self, hostname: str, ipv4: Optional[str], ipv6: Optional[str]
    ) -> None:
        services = self.ids.services
        services.data = [
            service for service in services.data if service["hostname"] != hostname
        ]
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
        if is_running:
            self.ids.task_indicator.text = "Handling action..."
        else:
            self.ids.task_indicator.text = ""

    def set_hostname(self, hostname: str) -> None:
        self.ids.hostname.text = hostname


class RecordingScreen(Screen):
    pass


class RecorderScreen(Screen):
    def add_recording(
        self, visible_name: str, uuid: str, recording: Recording, atEnd: bool = False
    ) -> None:
        recordings = self.ids.recordings
        recording_data = {"text": visible_name, "uuid": uuid, "recording": recording}
        if atEnd == True:
            recordings.data.append(recording_data)
        else:
            recordings.data.insert(0, recording_data)

    def remove_recording(self, uuid: str) -> None:
        recordings = self.ids.recordings
        recordings.data = [rec for rec in recordings.data if rec["uuid"] != uuid]

    def set_recording_status(self, is_recording: bool) -> None:
        if is_recording:
            self.ids.recorder_status.text = "Status: Recording"
        else:
            self.ids.recorder_status.text = "Status: Not recording"


# class LiveScreen(Screenhostname = "192.168.75.51"):
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
            circle_x = self.origin[0] + coord[0] * self.size[0]
            circle_y = self.origin[1] + (1 - coord[1]) * self.size[1]
            self.circle_obj = Line(circle=(circle_x, circle_y, GAZE_CIRCLE_RADIUS))
        self.canvas.add(self.circle_obj)
        self.canvas.remove(Color(1, 0, 0, 1))

    def reset(self):
        self.canvas.remove(self.circle_obj)
        self.circle_obj = Line(circle=(0, 0, 0))
        self.canvas.add(self.circle_obj)


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

        # MODIFICA: variabile per il timestamp dell'ultimo print del gaze
        self._last_gaze_print = 0.0

        # MODIFICA: variabili per YOLO
        self.yolo_model = None
        self._yolo_task = None
        self._last_yolo_infer = 0.0
        
        # MODIFICA: lista di detection correnti
        self._yolo_latest = []  

        # MODIFICA: per gestire up e down 
        self.history = deque(maxlen=HISTORY_LENGTH)
        self.state = 'stand_no_box'      # altri: 'bend_no_box', 'stand_with_box'
        self.last_action = None          # 'up' / 'down' / None
        self._last_action_time = 0.0

        # MODIFICA: per chiamare ROS2
        self.ros_node: Node | None = None
        self.traj_client = None
        self.adm_client = None
        self._ros_ready = False

        # MODIFICA: --- Box Gate & place-state ---
        self.box_gate = False

        # #MODIFICA: Baseline e logica relative-threshold
        self.jidx = None
        self.last_pos = None
        self.baseline_swB = None   # baseline all'ingresso in stand_with_box
        self.baseline_b2p = None   # baseline all'ingresso in bend_to_place
        self.above_since = None
        self.below_since = None
        self.arm_until = 0.0       # quando < now, i trigger sono armati


    def _init_ros(self):
        if self._ros_ready:
            return
        # init rclpy once
        if not rclpy.ok():
            rclpy.init()
        self.ros_node = rclpy.create_node('vision_command_node')
        self.traj_client = self.ros_node.create_client(RunTrajectory, RUN_TRAJ_SRV_NAME)
        self.adm_client  = self.ros_node.create_client(SetAdmittanceParams, SET_ADMITT_SRV_NAME)
        self.box_gate_pub = self.ros_node.create_publisher(Bool, 'perception/box_gate', 10)
        self.js_sub       = self.ros_node.create_subscription(JointState, 'joint_states', self._on_js, 30)

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

    def _send_trajectory_request_blocking(self, direction: str):
        # runs in a worker thread
        if not self._ros_ready or self.ros_node is None or self.traj_client is None:
            print("[ROS] traj client not ready")
            return
        req = RunTrajectory.Request()
        req.trajectory_type = direction
        future = self.traj_client.call_async(req)
        rclpy.spin_until_future_complete(self.ros_node, future)
        # Optionally handle future.result()

    def _set_admittance_params_blocking(self, k_val: float):
        # runs in a worker thread
        if not self._ros_ready or self.ros_node is None or self.adm_client is None:
            print("[ROS] adm client not ready")
            return
        req = SetAdmittanceParams.Request()
        req.k = k_val
        future = self.adm_client.call_async(req)
        rclpy.spin_until_future_complete(self.ros_node, future)

    async def _trigger_action_ros(self, action: str):
        """Asynchronously invoke both services without blocking the Kivy loop."""
        if not self._ros_ready:
            print("[ROS] not ready, skipping action")
            return
        k_val = K_DOWN if action == 'down' else K_UP
        # offload blocking rclpy calls
        await asyncio.to_thread(self._set_admittance_params_blocking, k_val)
        await asyncio.to_thread(self._send_trajectory_request_blocking, action)



    def _evaluate_state_and_print(self):
        n = len(self.history)
        if n == 0:
            return

        # --- FAST PATH opzionale: N consecutivi più rapidi ---
        if N_CONSEC_NOT_GRASPED > 0 and n >= N_CONSEC_NOT_GRASPED and self.state == 'stand_no_box':
            last_n = list(self.history)[-N_CONSEC_NOT_GRASPED:]
            if all(x == 'not_grasped' for x in last_n):
                action = 'down'
                self.state = 'bend_no_box'
                now = time.monotonic()
                if action != self.last_action or (now - self._last_action_time) >= ACTION_COOLDOWN_S:
                    print(f"[ACTION] {action.upper()}  | state -> {self.state}")
                    self.last_action = action
                    self._last_action_time = now
                    self.create_task(self._trigger_action_ros(action), name=f"ros_action_{action}_{int(now)}")
                return

        if N_CONSEC_GRASPED > 0 and n >= N_CONSEC_GRASPED and self.state == 'bend_no_box':
            last_n = list(self.history)[-N_CONSEC_GRASPED:]
            if all(x == 'grasped' for x in last_n):
                action = 'up'
                self.state = 'stand_with_box'
                # --- GATE ON SOLO QUI (UP da pick) ---
                self._publish_box_gate(True)
                # prepara la baseline per la fase place
                self.baseline_swB = None
                self.baseline_b2p = None
                self.arm_until = time.monotonic() + ARM_DELAY_S

                now = time.monotonic()
                if action != self.last_action or (now - self._last_action_time) >= ACTION_COOLDOWN_S:
                    print(f"[ACTION] {action.upper()}  | state -> {self.state}")
                    self.last_action = action
                    self._last_action_time = now
                    self.create_task(self._trigger_action_ros(action), name=f"ros_action_{action}_{int(now)}")
                return

        # --- Valutazione "lenta" con finestre/soglie diverse per stato ---
        if self.state == 'stand_no_box':
            k = min(TAIL_NOT_GRASPED, n)
            tail = list(self.history)[-k:]
            c = Counter(tail)
            r_not = c['not_grasped'] / k if k > 0 else 0.0

            if r_not >= THRESH_NOT_GRASPED:
                action = 'down'
                self.state = 'bend_no_box'
            else:
                action = None

        elif self.state == 'bend_no_box':
            k = min(TAIL_GRASPED, n)
            tail = list(self.history)[-k:]
            c = Counter(tail)
            r_grasp = c['grasped'] / k if k > 0 else 0.0

            if r_grasp >= THRESH_GRASPED:
                action = 'up'
                self.state = 'stand_with_box'
            else:
                action = None

        else:
            # 'stand_with_box' (nessuna azione per ora)
            action = None

        # Emissione azione con cooldown
        if action is not None:
            now = time.monotonic()
            if action != self.last_action or (now - self._last_action_time) >= ACTION_COOLDOWN_S:
                print(f"[ACTION] {action.upper()}  | state -> {self.state}")
                self.last_action = action
                self._last_action_time = now
                self.create_task(self._trigger_action_ros(action), name=f"ros_action_{action}_{int(now)}")


    def _publish_box_gate(self, val: bool):
        if not self._ros_ready:
            return
        if val != self.box_gate:
            self.box_gate = val
            self.box_gate_pub.publish(Bool(data=val))
            print(f"[BOX_GATE] → {val}")

    def _on_js(self, msg: JointState):
        # risolvo l’indice del giunto la prima volta
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

        # -------------- PLACE LOGIC (solo quando ho il box) --------------
        if self.state == 'stand_with_box':
            # baseline all'ingresso, dopo un piccolo arming delay
            if self.baseline_swB is None:
                if now >= self.arm_until:
                    self.baseline_swB = pos
                    print(f"[BASE] stand_with_box baseline_swB={pos:.3f} rad")
                return

            delta = pos - self.baseline_swB
            print(f"[Δ] stand_with_box: pos={pos:.3f} base={self.baseline_swB:.3f} Δ={delta:.3f} "
                f"(thr={DELTA_DOWN_PLACE:.3f})")
            if delta >= DELTA_DOWN_PLACE:
                # TRIGGER DOWN → bend_to_place
                action = 'down'
                print(f"[TRIGGER] stand_with_box → DOWN (place)")
                self.state = 'bend_to_place'
                # reset baseline del nuovo stato e arming
                self.baseline_b2p = None
                self.arm_until = now + ARM_DELAY_S
                # eseguo l’azione (K + service)
                self.create_task(self._trigger_action_ros(action), name=f"ros_action_{action}_{int(now)}")
                return

        elif self.state == 'bend_to_place':
            if self.baseline_b2p is None:
                if now >= self.arm_until:
                    self.baseline_b2p = pos
                    print(f"[BASE] bend_to_place baseline_b2p={pos:.3f} rad")
                return

            delta = self.baseline_b2p - pos
            print(f"[Δ] bend_to_place: pos={pos:.3f} base={self.baseline_b2p:.3f} Δ={delta:.3f} "
                f"(thr={DELTA_UP_PLACE:.3f})")
            if delta >= DELTA_UP_PLACE:
                # TRIGGER UP → stand_without_box
                action = 'up'
                print(f"[TRIGGER] bend_to_place → UP (return)")
                self.state = 'stand_without_box'
                # appena appoggio il box → gate OFF
                self._publish_box_gate(False)
                # pulizia
                self.baseline_swB = None
                self.baseline_b2p = None
                self.arm_until    = now + ARM_DELAY_S
                self.create_task(self._trigger_action_ros(action), name=f"ros_action_{action}_{int(now)}")
                return


    # MODIFICA: funzione per stampare il gaze nel terminale
    def _print_gaze(self, gaze_ts, gaze_data, rate_hz: float = 10.0):
        import time
        now = time.time()
        if now - self._last_gaze_print < 1.0 / rate_hz:
            return
        self._last_gaze_print = now
        try:
            pt = gaze_data["gaze2d"]
            print(f"[LIVE] gaze2d=({pt[0]:.3f}, {pt[1]:.3f}) ts={gaze_ts:.3f}s")
        except Exception as e:
            print(f"[LIVE] gaze missing/invalid: {e}")

    def build(self):
        return self

    def on_start(self):
        self.create_task(self.backend_app(), name="backend_app")
        self.send_app_event(AppEventKind.START_DISCOVERY)

    def close(self, *args) -> bool:
        self.send_app_event(AppEventKind.STOP)
        return True

    def switch_to_screen(self, screen: str):
        if screen == "discovery":
            self.transition.direction = "right"
        else:
            self.transition.direction = "left"
        self.current = screen

    def start_control(self) -> bool:
        selected = self.get_screen(
            "discovery"
        ).ids.services.ids.selectables.selected_nodes
        if len(selected) <= 0:
            popup = UserMessagePopup(title="No Glasses3 unit selected")
            popup.ids.message_label.text = (
                "Please select a Glasses3 unit and try again."
            )
            popup.open()
            return False
        else:
            # MODIFICA
            # hostname = self.get_screen("discovery").ids.services.data[selected[0]][
            #     "hostname"
            # ]
            hostname = "192.168.75.51"
            self.backend_control_task = self.create_task(
                self.backend_control(hostname), name="backend_control"
            )
            self.get_screen("control").set_hostname(hostname)
            self.switch_to_screen("control")
            return True

    async def stop_control(self) -> None:
        await self.cancel_task(self.backend_control_task)
        self.get_screen("control").clear()

    def start_discovery(self):
        self.discovery_task = self.create_task(
            self.backend_discovery(), name="backend_discovery"
        )
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
                self.get_screen("discovery").add_service(
                    service.hostname, service.ipv4_address, service.ipv6_address
                )
            case (EventKind.UPDATED, service):
                self.get_screen("discovery").update_service(
                    service.hostname, service.ipv4_address, service.ipv6_address
                )
            case (EventKind.REMOVED, service):
                self.get_screen("discovery").remove_service(
                    service.hostname, service.ipv4_address, service.ipv6_address
                )

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
                        await self.handle_control_event(
                            g3, await self.control_events.get()
                        )
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

    async def _yolo_loop(self, conf: float = 0.6, rate_hz: float = 3.0):
        """Legge l'ultimo frame disponibile, fa YOLO e aggiorna history/stato."""
        period = 1.0 / rate_hz
        while True:
            start = time.monotonic()
            try:
                fw = self.latest_frame_with_timestamp
                if not fw or fw[0] is None:
                    await asyncio.sleep(0.05)
                    continue

                # Converti a ndarray BGR (flip verticale come nel draw)
                frame = np.flip(fw[0].to_ndarray(format="bgr24"), 0)

                # Esegui inferenza in thread separato per non bloccare la UI
                results = await asyncio.to_thread(
                    self.yolo_model.predict, frame, conf=conf, verbose=False
                )
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

                # === QUI LA MODIFICA: salva detections e aggiorna history/stato ===
                self._yolo_latest = det_list

                # aggiorna history con 'grasped' / 'not_grasped' / 'none'
                detected_grasped = any(d["label"] == "Box-Grasped" for d in det_list)
                detected_not_grasped = any(d["label"] == "Box-Not Grasped" for d in det_list)

                if detected_grasped:
                    self.history.append('grasped')
                elif detected_not_grasped:
                    self.history.append('not_grasped')
                else:
                    self.history.append('none')

                # valuta macchina a stati e stampa UP/DOWN se scatta
                self._evaluate_state_and_print()
                # === FINE MODIFICA ===
                # processa i messaggi ROS (joint_states) senza bloccare
                if self._ros_ready:
                    try:
                        rclpy.spin_once(self.ros_node, timeout_sec=0.0)
                    except Exception:
                        pass


            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[YOLO] Error during inference: {e}")

            dt = time.monotonic() - start
            await asyncio.sleep(max(0.0, period - dt))



    def start_live_stream(self, g3: Glasses3) -> None:

        self._init_ros()

        async def live_stream():
            async with g3.stream_rtsp(scene_camera=True, gaze=True) as streams:
                async with streams.scene_camera.decode() as scene_stream, streams.gaze.decode() as gaze_stream:
                    # --- YOLO: load una volta quando parte il live ---
                    if self.yolo_model is None:
                        print(f"[YOLO] Loading model: {YOLO_MODEL_PATH}")
                        self.yolo_model = YOLO(YOLO_MODEL_PATH)
                        if torch.cuda.is_available():
                            print("[YOLO] Using CUDA")
                        else:
                            print("[YOLO] Using CPU")

                    # avvia il task di inferenza periodica
                    if self._yolo_task is None or self._yolo_task.done():
                        self._yolo_task = self.create_task(self._yolo_loop(conf=0.7, rate_hz=3.0), name="yolo_loop")
                    live_screen = self.get_screen("control").ids.sm.get_screen("live")
                    Window.bind(on_resize=live_screen.clear)
                    self.latest_frame_with_timestamp = await scene_stream.get()
                    self.latest_gaze_with_timestamp = await gaze_stream.get()
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
                    self.draw_frame_event = Clock.schedule_interval(
                        draw_frame, 1 / LIVE_FRAME_RATE
                    )
                    await self.read_frames_task

        async def update_frame(scene_stream, gaze_stream, streams):
            while True:
                latest_frame_with_timestamp = await scene_stream.get()
                latest_gaze_with_timestamp = await gaze_stream.get()
                while (
                    latest_gaze_with_timestamp[1] is None
                    or latest_frame_with_timestamp[1] is None
                ):
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
            if (
                self.latest_frame_with_timestamp is None
                or self.latest_gaze_with_timestamp is None
                or self.live_gaze_circle is None
            ):
                logging.warning(
                    "Frame not drawn due to missing frame, gaze data or gaze circle."
                )
                return
            display = self.get_screen("control").ids.sm.get_screen("live").ids.display
            # image = np.flip(
            #     self.latest_frame_with_timestamp[0].to_ndarray(format="bgr24"), 0
            # )

            # frame -> BGR + flip verticale
            image = np.flip(self.latest_frame_with_timestamp[0].to_ndarray(format="bgr24"), 0)

            # OpenCV vuole memoria contigua (no stride negativo)
            if not image.flags["C_CONTIGUOUS"]:
                image = np.ascontiguousarray(image)   # oppure: image = image.copy()


            # --- Overlay YOLO boxes e label ---
            h, w = image.shape[:2]
            dets = getattr(self, "_yolo_latest", [])
            for d in dets:
                x1, y1, x2, y2 = d["xyxy"]
                # clip in-range per evitare errori se YOLO sfora il bordo
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


            texture = Texture.create(
                size=(image.shape[1], image.shape[0]), colorfmt="bgr"
            )
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
            if self.live_stream_task is not None:
                return not self.live_stream_task.done()
            else:
                return False

        if live_stream_task_running():
            logging.info("Task not started: live_stream_task already running.")
        else:
            self.live_stream_task = self.create_task(
                live_stream(), name="live_stream_task"
            )

    async def stop_live_stream(self) -> None:
        if self.read_frames_task is not None:
            if not self.read_frames_task.cancelled():
                await self.cancel_task(self.read_frames_task)
        if self.live_stream_task is not None:
            if not self.live_stream_task.cancelled():
                await self.cancel_task(self.live_stream_task)
        if self.draw_frame_event is not None:
            self.draw_frame_event.cancel()
            self.draw_frame_event = None
        live_screen = self.get_screen("control").ids.sm.get_screen("live")
        Window.unbind(on_resize=live_screen.clear)
        live_screen.clear()
        self.last_texture = None
        # MODIFICA: ROS
        self._shutdown_ros()

    def get_selected_recording(self) -> Optional[str]:
        recordings = (
            self.get_screen("control").ids.sm.get_screen("recorder").ids.recordings
        )
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
            videoplayer = (
                self.get_screen("control")
                .ids.sm.get_screen("recording")
                .ids.videoplayer
            )
            videoplayer.source = file_url
            videoplayer.state = "play"

            async with aiohttp.ClientSession() as session:
                async with session.get(await recording.get_gazedata_url()) as response:
                    all_gaze_data = await response.text()
            gaze_json_list = all_gaze_data.split("\n")[:-1]
            self.gaze_data_list = []
            for gaze_json in gaze_json_list:
                self.gaze_data_list.append(json.loads(gaze_json))

            if self.replay_gaze_circle is None:
                video_height = videoplayer.size[0] * VIDEO_Y_TO_X_RATIO
                video_origin_y = (
                    videoplayer.size[1] - video_height + VIDEOPLAYER_PROGRESS_BAR_HEIGHT
                ) / 2
                self.replay_gaze_circle = GazeCircle(
                    videoplayer.canvas,
                    (0, video_origin_y),
                    (videoplayer.size[0], video_height),
                )
                self.bind_replay_gaze_updates()

    def bind_replay_gaze_updates(self):
        def reset_gaze_circle(instance, state):
            if state == "start" or state == "stop":
                if self.replay_gaze_circle is not None:
                    self.replay_gaze_circle.reset()

        def update_gaze_circle(instance, timestamp):
            if self.replay_gaze_circle is None:
                logging.warning("Gaze not drawn due to missing gaze circle.")
                return
            current_gaze_index = self.binary_search_gaze_point(
                timestamp, self.gaze_data_list
            )
            try:
                point = self.gaze_data_list[current_gaze_index]["data"]["gaze2d"]
            except KeyError:
                point = None
            self.replay_gaze_circle.redraw(point)

        videoplayer = (
            self.get_screen("control").ids.sm.get_screen("recording").ids.videoplayer
        )
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
            if abs(gaze_list[mid_index]["timestamp"] - value) < abs(
                gaze_list[best_index]["timestamp"] - value
            ):
                best_index = mid_index
        return best_index

    async def delete_selected_recording(self, g3: Glasses3) -> None:
        uuid = self.get_selected_recording()
        if uuid is not None:
            await g3.recordings.delete(uuid)

    async def update_recordings(self, g3, recordings_events):
        recorder_screen = self.get_screen("control").ids.sm.get_screen("recorder")
        for child in cast(List[Recording], g3.recordings):
            recorder_screen.add_recording(
                await child.get_visible_name(), child.uuid, child, atEnd=True
            )
        while True:
            event = await recordings_events.get()
            match event:
                case (RecordingsEventKind.ADDED, body):
                    uuid = cast(List[str], body)[0]
                    recording = g3.recordings.get_recording(uuid)
                    recorder_screen.add_recording(
                        await recording.get_visible_name(), recording.uuid, recording
                    )
                case (RecordingsEventKind.REMOVED, body):
                    uuid = cast(List[str], body)[0]
                    recorder_screen.remove_recording(uuid)

    async def start_update_recorder_status(self, g3: Glasses3) -> None:
        recorder_screen = self.get_screen("control").ids.sm.get_screen("recorder")
        if await g3.recorder.get_created() != None:
            recorder_screen.set_recording_status(True)
        else:
            recorder_screen.set_recording_status(False)
        (
            recorder_started_queue,
            self.unsubscribe_to_recorder_started,
        ) = await g3.recorder.subscribe_to_started()
        (
            recorder_stopped_queue,
            self.unsubscribe_to_recorder_stopped,
        ) = await g3.recorder.subscribe_to_stopped()

        async def handle_recorder_started(
            recorder_started_queue: asyncio.Queue[SignalBody],
        ):
            while True:
                await recorder_started_queue.get()
                recorder_screen.set_recording_status(True)

        async def handle_recorder_stopped(
            recorder_stopped_queue: asyncio.Queue[SignalBody],
        ):
            while True:
                await recorder_stopped_queue.get()
                recorder_screen.set_recording_status(False)

        self.handle_recorder_started_task = self.create_task(
            handle_recorder_started(recorder_started_queue),
            name="handle_recorder_started",
        )
        self.handle_recorder_stopped_task = self.create_task(
            handle_recorder_stopped(recorder_stopped_queue),
            name="handle_recorder_stopped",
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
