#!/usr/bin/env python3
# fsm_pick_place_with_yolo_closure.py
#
# PICK (YOLO):
#   stand_without_box --(Box-Not Grasped stabile)--> action='down'  -> bend_to_pick
#   bend_to_pick      --(Box-Grasped    stabile)--> action='up'    -> stand_with_box  [publish_box_gate(True)]
#
# PLACE (JointState, soglie relative):
#   stand_with_box  --(Δ>=DELTA_DOWN_PLACE)--> action='down' -> bend_to_place
#   bend_to_place   --(Δ>=DELTA_UP_PLACE)  --> action='up'   -> stand_without_box [publish_box_gate(False)]
#
# Azioni inviate con le stesse helper del tuo snippet:
#   publish_box_gate(...)
#   set_admittance_params(...)
#   send_trajectory_request(...)

import os
import time
from collections import deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from exo_interfaces.srv import RunTrajectory, SetAdmittanceParams

# -------- YOLO + RealSense --------
YOLO_ENABLED    = True
YOLO_MODEL_PATH = 'src/exo_control/yolo_weights/best-realsense.pt'
CONF_THRESHOLD  = 0.6

import cv2
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs

# -------- Hysteresis YOLO --------
TRIGGER_PERCENTAGE_NOT_GRASPED = 0.9
TRIGGER_PERCENTAGE_GRASPED     = 0.6
HISTORY_LENGTH_NOT_GRASPED     = 80
HISTORY_LENGTH_GRASPED         = 60

# -------- K target --------
K_DOWN = 30.0
K_UP   = 30.0

# -------- Soglie relative (place) --------
DELTA_DOWN_PLACE = 0.04  # stand_with_box -> bend_to_place: pos - baseline_swB >= ...
DELTA_UP_PLACE   = 0.05  # bend_to_place -> stand_without_box: baseline_b2p - pos >= ...
HOLD_DOWN_S      = 0.0
HOLD_UP_S        = 0.0
ARM_DELAY_S      = 3.0   # piccolo delay per evitare rimbalzi dopo cambio stato

# -------- Joint da monitorare --------
JOINT_NAME     = 'joint_0'
FALLBACK_INDEX = 0

def main():
    # ---------------- ROS init ----------------
    rclpy.init()
    node = rclpy.create_node('pick_place_fsm')

    # --- Service clients
    traj_client = node.create_client(RunTrajectory, 'run_trajectory')
    adm_client  = node.create_client(SetAdmittanceParams, 'set_admittance_params')
    have_traj = traj_client.wait_for_service(timeout_sec=1.0)
    have_adm  = adm_client.wait_for_service(timeout_sec=1.0)
    if not (have_traj and have_adm):
        node.get_logger().warn("ROS services not available yet. Actions will be skipped until services appear.")

    # --- Publisher gate
    box_gate_pub = node.create_publisher(Bool, 'perception/box_gate', 10)
    box_gate = False  # latched

    # ---------------- Helper EXACT come tuo snippet ----------------
    def send_trajectory_request(direction: str):
        if not traj_client.service_is_ready():
            node.get_logger().warn("run_trajectory not ready; skipping command.")
            return
        req = RunTrajectory.Request()
        req.trajectory_type = direction
        fut = traj_client.call_async(req)
        # NON bloccare: solo log al termine
        fut.add_done_callback(lambda f: node.get_logger().info(
            f"[run_trajectory] '{direction}' → "
            f"{'ok' if (f.exception() is None) else 'err'}"))

    def set_admittance_params(k_val: float):
        if not adm_client.service_is_ready():
            node.get_logger().warn("set_admittance_params not ready; skipping.")
            return
        req = SetAdmittanceParams.Request()
        req.k = k_val
        fut = adm_client.call_async(req)
        # NON bloccare: solo log al termine
        def _done(f):
            msg = ""
            try:
                res = f.result()
                msg = getattr(res, 'message', '')
            except Exception as e:
                msg = f"err: {e}"
            node.get_logger().info(f"[set_admittance_params] K={k_val} → {msg}")
        fut.add_done_callback(_done)

    def publish_box_gate(new_val: bool):
        nonlocal box_gate
        if new_val != box_gate:
            box_gate = new_val
            box_gate_pub.publish(Bool(data=box_gate))
            node.get_logger().info(f"[BOX_GATE] → {box_gate}")


    # ---------------- Stato FSM ----------------
    state = 'stand_without_box'   # -> bend_to_pick -> stand_with_box -> bend_to_place -> stand_without_box
    arm_until = time.monotonic() + ARM_DELAY_S

    # ---------------- Stato place (relative thresholds) ----------------
    jidx = None
    last_pos = None
    baseline_swB = None   # baseline in stand_with_box
    baseline_b2p = None   # baseline in bend_to_place
    above_since = None
    below_since = None

    # ---------------- YOLO+RealSense ----------------
    history_not_gr = deque(maxlen=HISTORY_LENGTH_NOT_GRASPED)
    history_gr     = deque(maxlen=HISTORY_LENGTH_GRASPED)
    have_display   = bool(os.environ.get("DISPLAY"))
    yolo_ok = False
    if YOLO_ENABLED:
        try:
            model = YOLO(YOLO_MODEL_PATH)
            pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(cfg)
            yolo_ok = True
            if have_display:
                cv2.namedWindow("YOLO RealSense Stream", cv2.WINDOW_NORMAL)
            node.get_logger().info("YOLO/RealSense ready.")
        except Exception as e:
            node.get_logger().error(f"Cannot start YOLO/RealSense: {e}")
            yolo_ok = False
            pipeline = None
            model = None
    else:
        model = None
        pipeline = None

    # ---------------- JointState callback ----------------
    def joint_state_cb(msg: JointState):
        nonlocal jidx, last_pos, baseline_swB, baseline_b2p, above_since, below_since, arm_until, state

        if not msg.position:
            return
        if jidx is None:
            if msg.name and JOINT_NAME in msg.name:
                jidx = msg.name.index(JOINT_NAME)
            else:
                jidx = min(FALLBACK_INDEX, len(msg.position)-1)
            node.get_logger().info(f"[MEAS READY] Using joint index {jidx} "
                                   f"({'name:'+JOINT_NAME if msg.name else 'no names'})")

        if jidx >= len(msg.position):
            return

        pos = float(msg.position[jidx])
        last_pos = pos

        # rispetta arming delay
        if time.monotonic() < arm_until:
            return

        # ----- PLACE logic: relative thresholds -----
        if state == 'stand_with_box':
            if baseline_swB is None:
                baseline_swB = pos
                node.get_logger().info(f"[STATE] stand_with_box | baseline={baseline_swB:.3f} rad (arming {ARM_DELAY_S:.2f}s)")
                arm_until = time.monotonic() + ARM_DELAY_S
                return

            delta = pos - baseline_swB
            node.get_logger().info(f"[DBG] stand_with_box Δ={delta:.3f} (>= {DELTA_DOWN_PLACE:.3f} ?)")
            if delta >= DELTA_DOWN_PLACE:
                if HOLD_DOWN_S <= 0.0 or _hold_ok('above'):
                    action = 'down'
                    publish_box_gate(True)  # si continua a trasportare, il gate resta ON
                    set_admittance_params(K_DOWN if action == 'down' else K_UP)
                    send_trajectory_request(action)

                    state = 'bend_to_place'
                    arm_until = time.monotonic() + ARM_DELAY_S
                    baseline_b2p = None
                    _reset_hold()

        elif state == 'bend_to_place':
            if baseline_b2p is None:
                baseline_b2p = pos
                node.get_logger().info(f"[STATE] bend_to_place | baseline={baseline_b2p:.3f} rad (arming {ARM_DELAY_S:.2f}s)")
                arm_until = time.monotonic() + ARM_DELAY_S
                return

            delta = baseline_b2p - pos
            node.get_logger().info(f"[DBG] bend_to_place Δ={delta:.3f} (>= {DELTA_UP_PLACE:.3f} ?)")
            if delta >= DELTA_UP_PLACE:
                if HOLD_UP_S <= 0.0 or _hold_ok('below'):
                    action = 'up'
                    publish_box_gate(False)  # rilasciato/appoggiato → gate OFF
                    set_admittance_params(K_DOWN if action == 'down' else K_UP)
                    send_trajectory_request(action)

                    state = 'stand_without_box'
                    arm_until = time.monotonic() + ARM_DELAY_S
                    baseline_swB = None
                    baseline_b2p = None
                    _reset_hold()

    # Hold helpers (closure)
    def _hold_ok(which: str) -> bool:
        nonlocal above_since, below_since
        now = time.monotonic()
        if which == 'above':
            if above_since is None:
                above_since = now
                return False
            return (now - above_since) >= HOLD_DOWN_S
        else:
            if below_since is None:
                below_since = now
                return False
            return (now - below_since) >= HOLD_UP_S

    def _reset_hold():
        nonlocal above_since, below_since
        above_since = None
        below_since = None

    # subscribe
    node.create_subscription(JointState, 'joint_states', joint_state_cb, 30)

    node.get_logger().info("Starting loop… (Ctrl+C per uscire)")
    try:
        while rclpy.ok():
            # serve gli eventi ROS (joint_states)
            rclpy.spin_once(node, timeout_sec=0.0)

            # --- YOLO step per fasi di PICK ---
            if yolo_ok and (time.monotonic() >= arm_until):
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame = np.asanyarray(color_frame.get_data())

                # Evita 'Unsupported image type': usa la forma callable su np.ndarray
                results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
                boxes = results[0].boxes if results else None

                detected_grasped = False
                detected_not_grasped = False

                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf   = float(box.conf[0])
                        if conf < CONF_THRESHOLD:
                            continue
                        label = model.names[cls_id]
                        if label == 'Box-Grasped':
                            detected_grasped = True
                        elif label == 'Box-Not Grasped':
                            detected_not_grasped = True

                        # overlay
                        if have_display:
                            xyxy = box.xyxy[0].cpu().numpy().astype(int)
                            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0,255,0), 2)
                            cv2.putText(frame, f"{label} {conf:.2f}", (xyxy[0], xyxy[1]-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                # aggiorna hysteresis
                history_not_gr.append(1 if detected_not_grasped else 0)
                history_gr.append(1 if detected_grasped else 0)
                MIN_NOT = max(10, HISTORY_LENGTH_NOT_GRASPED // 4)
                MIN_GR  = max(10, HISTORY_LENGTH_GRASPED // 4)
                pct_not = (sum(history_not_gr)/len(history_not_gr)) if history_not_gr else 0.0
                pct_gr  = (sum(history_gr)/len(history_gr))         if history_gr     else 0.0

                # Transizioni PICK
                if state == 'stand_without_box':
                    if len(history_not_gr) >= MIN_NOT and pct_not >= TRIGGER_PERCENTAGE_NOT_GRASPED:
                        action = 'down'
                        # NB: durante pick il gate rimane OFF; si accende solo quando sali col box
                        set_admittance_params(K_DOWN if action == 'down' else K_UP)
                        send_trajectory_request(action)
                        state = 'bend_to_pick'
                        arm_until = time.monotonic() + ARM_DELAY_S
                        history_gr.clear()

                elif state == 'bend_to_pick':
                    if len(history_gr) >= MIN_GR and pct_gr >= TRIGGER_PERCENTAGE_GRASPED:
                        action = 'up'
                        publish_box_gate(True)  # SOLO qui: gate ON quando sali col box
                        set_admittance_params(K_DOWN if action == 'down' else K_UP)
                        send_trajectory_request(action)
                        state = 'stand_with_box'
                        arm_until = time.monotonic() + ARM_DELAY_S
                        # baseline_swB verrà acquisita nel JointState
                        history_not_gr.clear()

                # overlay
                if have_display:
                    cv2.putText(frame, f"STATE: {state}", (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    cv2.putText(frame, f"not_gr={pct_not:.2f} ({len(history_not_gr)})", (10, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                    cv2.putText(frame, f"grasped={pct_gr:.2f} ({len(history_gr)})", (10, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                    cv2.imshow("YOLO RealSense Stream", frame)
                    if cv2.waitKey(1) == ord('q'):
                        break

            else:
                # se YOLO è disabilitato o in arming delay, piccolo sleep
                time.sleep(0.005)

    except KeyboardInterrupt:
        pass
    finally:
        if YOLO_ENABLED and yolo_ok:
            try:
                pipeline.stop()
            except Exception:
                pass
            if have_display:
                cv2.destroyAllWindows()
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()
        print("System shutdown complete.")

if __name__ == '__main__':
    main()
