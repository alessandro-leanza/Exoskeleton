#!/usr/bin/env python3
# exo_g3pycontroller_mic_ros.py
#
# Comandi vocali "UP/DOWN" (Vosk) + FSM pick/place con:
# - box_gate ON/OFF
# - place automatico con baseline relative sui JointState
# - arming delay per evitare baseline premature

import os
import time
import queue
import json
import argparse
from typing import Optional

import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer

# ROS2
import rclpy
from rclpy.node import Node
from exo_interfaces.srv import RunTrajectory, SetAdmittanceParams
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState

# ---------- Config/keywords ----------
KEYWORDS = {"up", "down"}            # parole chiave
DEBOUNCE_S = 1.0                     # evita ripetizioni troppo ravvicinate

# FSM + place (delta relativi)
DELTA_DOWN_PLACE  = 0.04   # stand_with_box -> (DOWN) -> bend_to_place: pos - base_swB >= soglia
DELTA_UP_PLACE    = 0.05   # bend_to_place   -> (UP)   -> stand_no_box:  base_b2p - pos >= soglia
ARM_DELAY_S       = 3.2   # ritardo prima di fissare la baseline dopo un cambio stato

# Admittance e servizi
RUN_TRAJ_SRV_NAME = "run_trajectory"
SET_ADMITT_SRV_NAME = "set_admittance_params"
# K_UP_DEFAULT   = 30.0
# K_DOWN_DEFAULT = 30.0
K_UP_DEFAULT   = 0.0
K_DOWN_DEFAULT = 0.0

# Joint da monitorare
JOINT_NAME     = "joint_0"
FALLBACK_INDEX = 0


# ------- Utility audio --------
def find_device_id(target_substr: Optional[str]):
    """Trova un device input che contiene 'target_substr' nel nome. None => default."""
    if not target_substr:
        return None
    devices = sd.query_devices()
    for idx, d in enumerate(devices):
        name = d.get("name", "")
        if target_substr.lower() in name.lower():
            return idx
    return None


# ------- ROS clients helper --------
class RosClients:
    def __init__(self, node: Node, traj_srv: str, adm_srv: str):
        self.node = node
        self.traj_client = node.create_client(RunTrajectory, traj_srv)
        self.adm_client  = node.create_client(SetAdmittanceParams, adm_srv)

    def wait_for_services(self, timeout_sec: float = 5.0):
        ok1 = self.traj_client.wait_for_service(timeout_sec=timeout_sec)
        ok2 = self.adm_client.wait_for_service(timeout_sec=timeout_sec)
        if not ok1:
            print(f"[ROS] WARNING: service not available: {RUN_TRAJ_SRV_NAME}")
        if not ok2:
            print(f"[ROS] WARNING: service not available: {SET_ADMITT_SRV_NAME}")
        return ok1 and ok2

    def send_trajectory(self, direction: str):
        req = RunTrajectory.Request()
        req.trajectory_type = direction
        fut = self.traj_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, fut)
        if fut.result() is None:
            print(f"[ROS] run_trajectory({direction}) FAILED")
        else:
            print(f"[ROS] run_trajectory({direction}) OK")

    def set_admittance(self, k_val: float):
        req = SetAdmittanceParams.Request()
        req.k = k_val
        fut = self.adm_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, fut)
        if fut.result() is None:
            print(f"[ROS] set_admittance_params(k={k_val}) FAILED")
        else:
            print(f"[ROS] set_admittance_params(k={k_val}) OK")
    
    def set_admittance_async(self, k_val: float):
        req = SetAdmittanceParams.Request()
        req.k = k_val
        fut = self.adm_client.call_async(req)
        def _done(f):
            try:
                _ = f.result()
                print(f"[ROS] set_admittance_params(k={k_val}) OK")
            except Exception as e:
                print(f"[ROS] set_admittance_params(k={k_val}) FAILED: {e}")
        fut.add_done_callback(_done)
        return fut  # per chain

    def send_trajectory_async(self, direction: str):
        req = RunTrajectory.Request()
        req.trajectory_type = direction
        fut = self.traj_client.call_async(req)
        def _done(f):
            try:
                _ = f.result()
                print(f"[ROS] run_trajectory({direction}) OK")
            except Exception as e:
                print(f"[ROS] run_trajectory({direction}) FAILED: {e}")
        fut.add_done_callback(_done)
        return fut


# ------- FSM + Place logic --------
class MicFSM:
    """
    Stati:
      - 'stand_no_box'
      - 'bend_no_box'
      - 'stand_with_box'
      - 'bend_to_place'
    """
    def __init__(self, node: Node, ros: RosClients, k_up: float, k_down: float):
        self.node = node
        self.ros = ros
        self.K_UP = k_up
        self.K_DOWN = k_down

        # Publisher gate
        self.box_gate_pub = node.create_publisher(Bool, "perception/box_gate", 10)
        self.box_gate = False
        self._publish_box_gate(False)  # stato iniziale

        # Sub JointState
        self.jidx = None
        self.last_pos = None
        self.js_sub = node.create_subscription(JointState, "joint_states", self._on_js, 30)

        # FSM
        self.state = "stand_no_box"
        self.arm_until = 0.0

        # Baseline place
        self.baseline_swB = None  # all'ingresso in stand_with_box
        self.baseline_b2p = None  # all'ingresso in bend_to_place

        # Debounce lato voce (ulteriore safety, oltre al main)
        self._last_cmd = None
        self._last_cmd_t = 0.0

        self._last_delta_log = 0.0
        self._delta_log_period = 0.25 

        print(f"[FSM] init: state={self.state}, gate={self.box_gate}")

    # ---------- Helpers ----------
    def _publish_box_gate(self, val: bool):
        if val != self.box_gate:
            self.box_gate = val
            self.box_gate_pub.publish(Bool(data=val))
            print(f"[BOX_GATE] → {val}")

    # def _do_action(self, direction: str):
    #     """Invia K + traiettoria."""
    #     if direction == "up":
    #         self.ros.set_admittance(self.K_UP)
    #     else:
    #         self.ros.set_admittance(self.K_DOWN)
    #     self.ros.send_trajectory(direction)

    def _do_action(self, direction: str):
        k = self.K_UP if direction == "up" else self.K_DOWN
        # 1) set admittance (async), 2) quando finisce → trajectory (async)
        # adm_fut = self.ros.set_admittance_async(k)
        # def _after_adm(_):
        #     self.ros.send_trajectory_async(direction)
        # adm_fut.add_done_callback(_after_adm)
        self.ros.set_admittance_async(k)
        self.ros.send_trajectory_async(direction)

    # ---------- Voice handler ----------
    def handle_voice(self, cmd: str, debounce_s: float):
        now = time.monotonic()
        if (cmd == self._last_cmd) and ((now - self._last_cmd_t) < debounce_s):
            return
        self._last_cmd = cmd
        self._last_cmd_t = now

        print(f"[VOICE] {cmd.upper()} | state={self.state}")

        # PICK PHASE (voce gestisce pick)
        if cmd == "down":
            if self.state == "stand_no_box":
                # Inizio piegamento per prendere
                self._do_action("down")
                self.state = "bend_no_box"
                print(f"[STATE] stand_no_box -> bend_no_box")
            elif self.state == "stand_with_box":
                # place è automatico via delta: ignoro il "down" vocale per coerenza
                print("[VOICE] 'down' ignorato: il place è automatico (delta JointState).")
            else:
                print("[VOICE] 'down' non applicabile in questo stato.")
            return

        if cmd == "up":
            if self.state == "bend_no_box":
                # Fine pick: salgo con BOX preso
                self._do_action("up")
                self.state = "stand_with_box"
                self._publish_box_gate(True)
                # prepara baseline per place con arming delay
                self.baseline_swB = None
                self.baseline_b2p = None
                self.arm_until = time.monotonic() + ARM_DELAY_S
                print(f"[STATE] bend_no_box -> stand_with_box | arm {ARM_DELAY_S:.2f}s")
            elif self.state == "bend_to_place":
                # override manuale di ritorno (se proprio vuoi)
                self._do_action("up")
                self.state = "stand_no_box"
                self._publish_box_gate(False)
                self.baseline_swB = None
                self.baseline_b2p = None
                self.arm_until = time.monotonic() + ARM_DELAY_S
                print(f"[STATE] bend_to_place -> stand_no_box (manual override) | arm {ARM_DELAY_S:.2f}s")
            else:
                print("[VOICE] 'up' non applicabile in questo stato.")
            return

    # ---------- JointState callbacks (PLACE automatico) ----------
    def _on_js(self, msg: JointState):
        # risolvo indice giunto al primo giro
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

        # ---- stand_with_box: aspetta arming, poi baseline e controlla Δ DOWN ----
        if self.state == "stand_with_box":
            if self.baseline_swB is None:
                if now < self.arm_until:
                    # in attesa per evitare baseline premature
                    return
                self.baseline_swB = pos
                print(f"[BASE] stand_with_box baseline_swB={pos:.3f} rad")
                return

            delta = pos - self.baseline_swB
            now = time.monotonic()
            if now - self._last_delta_log >= self._delta_log_period:
                self._last_delta_log = now
                print(f"[Δ] swB: pos={pos:.3f} base={self.baseline_swB:.3f} Δ={delta:.3f} "
                    f"(thr={DELTA_DOWN_PLACE:.3f})")

            if delta >= DELTA_DOWN_PLACE:
                print(f"[TRIGGER] stand_with_box → DOWN (place)")
                self._do_action("down")
                self.state = "bend_to_place"
                self.baseline_b2p = None
                self.arm_until = now + ARM_DELAY_S
                return

        # ---- bend_to_place: aspetta arming, baseline e controlla Δ UP ----
        elif self.state == "bend_to_place":
            if self.baseline_b2p is None:
                if now < self.arm_until:
                    return
                self.baseline_b2p = pos
                print(f"[BASE] bend_to_place baseline_b2p={pos:.3f} rad")
                return

            delta = self.baseline_b2p - pos
            now = time.monotonic()
            if now - self._last_delta_log >= self._delta_log_period:
                self._last_delta_log = now
                print(f"[Δ] b2p: pos={pos:.3f} base={self.baseline_b2p:.3f} Δ={delta:.3f} "
                    f"(thr={DELTA_UP_PLACE:.3f})")

            if delta >= DELTA_UP_PLACE:
                print(f"[TRIGGER] bend_to_place → UP (return)")
                self._do_action("up")
                self.state = "stand_no_box"
                self._publish_box_gate(False)
                self.baseline_swB = None
                self.baseline_b2p = None
                self.arm_until = now + ARM_DELAY_S
                return


# ------- Main -------
def main():
    ap = argparse.ArgumentParser(description="ASR (UP/DOWN) con Vosk + FSM place/box_gate + trigger ROS2")
    ap.add_argument("--model", type=str, default="/home/alessandro/exo_v2_ws/models/vosk-model-small-en-us-0.15",
                    help="Percorso al modello Vosk")
    ap.add_argument("--device", type=str, default=None,
                    help="Substring del nome dispositivo input (es. 'Samsung' / 'Bluetooth'). None=default")
    ap.add_argument("--rate", type=int, default=None,
                    help="Forza sample rate (Hz). Default: usa quello di default del device.")
    ap.add_argument("--list-devices", action="store_true", help="Stampa i device audio e esci")
    ap.add_argument("--traj-service", type=str, default=RUN_TRAJ_SRV_NAME,
                    help=f"Nome service ROS2 per la traiettoria (default: {RUN_TRAJ_SRV_NAME})")
    ap.add_argument("--adm-service", type=str, default=SET_ADMITT_SRV_NAME,
                    help=f"Nome service ROS2 per l'admittance (default: {SET_ADMITT_SRV_NAME})")
    ap.add_argument("--k-up", type=float, default=K_UP_DEFAULT, help="K per UP")
    ap.add_argument("--k-down", type=float, default=K_DOWN_DEFAULT, help="K per DOWN")
    ap.add_argument("--no-ros", action="store_true", help="Non inviare comandi ROS, solo stampa")
    ap.add_argument("--debounce", type=float, default=DEBOUNCE_S, help="Debounce comandi (s)")
    ap.add_argument("--blocksize", type=int, default=2048, help="Dimensione blocco audio (campioni). Default: 2048")
    args = ap.parse_args()

    if args.list_devices:
        devices = sd.query_devices()
        default_in = sd.default.device[0]
        for i, d in enumerate(devices):
            star = "*" if i == default_in else " "
            print(f"{star} {i:2d} {d['name']}, {d['hostapi']} "
                  f"({int(d['max_input_channels'])} in, {int(d['max_output_channels'])} out)")
        return

    if not os.path.isdir(args.model):
        raise FileNotFoundError(
            f"Modello Vosk non trovato in {args.model}. Scaricalo e imposta --model."
        )

    print("[MIC] Loading Vosk model…")
    model = Model(args.model)

    # Selezione device
    device_id = find_device_id(args.device)
    if device_id is not None:
        print(f"[MIC] Using device id {device_id} matching '{args.device}'")
        device_info = sd.query_devices(device_id, 'input')
    else:
        print("[MIC] Using default input device")
        default_in = sd.default.device[0]
        device_info = sd.query_devices(default_in, 'input')

    # Sample rate
    sample_rate = int(args.rate) if args.rate else int(device_info['default_samplerate'])
    # blocksize = max(1024, int(sample_rate // 2))  # ~0.5s
    blocksize = int(args.blocksize)
    print(f"[MIC] SR={sample_rate} Hz, blocksize={blocksize}")

    # Inizializza Vosk
    rec = KaldiRecognizer(model, sample_rate)
    rec.SetGrammar(json.dumps(list(KEYWORDS)))  # lessico ristretto

    # ROS (opzionale)
    fsm = None
    if not args.no_ros:
        if not rclpy.ok():
            rclpy.init()
        node = rclpy.create_node("voice_command_node")
        ros = RosClients(node, args.traj_service, args.adm_service)
        ros.wait_for_services(timeout_sec=5.0)
        fsm = MicFSM(node, ros, k_up=args.k_up, k_down=args.k_down)
    else:
        node = None

    audio_q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[MIC] Status: {status}", flush=True)
        mono = indata
        if mono.ndim > 1:
            mono = np.mean(mono, axis=1)
        pcm16 = (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        audio_q.put(pcm16)

    last_cmd = None
    last_time = 0.0
    debounce_s = float(args.debounce)

    print("[MIC] Start listening. Say 'UP' or 'DOWN'. Ctrl+C to exit.")
    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32",
                            blocksize=blocksize, callback=callback, device=device_id):
            while True:
                data = audio_q.get()
                text = ""

                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    text = (res.get("text") or "").strip().lower()
                    if text:
                        print(f"[ASR:final] {text}")
                else:
                    res = json.loads(rec.PartialResult())
                    text = (res.get("partial") or "").strip().lower()
                    if text:
                        print(f"[ASR:partial] {text}")

                if not text:
                    # comunque processa ROS callbacks
                    if node is not None:
                        rclpy.spin_once(node, timeout_sec=0.0)
                    continue

                words = set(text.split())
                hit = "up" if "up" in words else ("down" if "down" in words else None)
                if not hit:
                    if node is not None:
                        rclpy.spin_once(node, timeout_sec=0.0)
                    continue

                now = time.monotonic()
                if (hit != last_cmd) or ((now - last_time) >= debounce_s):
                    last_cmd = hit
                    last_time = now

                    if fsm is not None:
                        fsm.handle_voice(hit, debounce_s)
                    else:
                        # modalità no-ros: solo stampa
                        print(f"[VOICE] {hit.upper()} (no-ros)")

                # Processa JointState/ROS
                if node is not None:
                    rclpy.spin_once(node, timeout_sec=0.0)

    except KeyboardInterrupt:
        print("\n[MIC] Stopped.")
    finally:
        if node is not None and rclpy.ok():
            try:
                node.destroy_node()
            except Exception:
                pass
            rclpy.shutdown()


if __name__ == "__main__":
    main()
