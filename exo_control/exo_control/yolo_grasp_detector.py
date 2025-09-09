#!/usr/bin/env python3
# yolo_grasp_detector.py

import os
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from exo_interfaces.srv import RunTrajectory, SetAdmittanceParams

# =======================
# Parametri principali
# =======================
YOLO_MODEL_PATH = 'src/exo_control/yolo_weights/best-realsense.pt'

# Soglie inferenza/detection
CONF_THRESHOLD = 0.6

# Hysteresis semplice sulle due classi (finestre differenti)
TRIGGER_PERCENTAGE_NOT_GRASPED = 0.9
TRIGGER_PERCENTAGE_GRASPED = 0.6
HISTORY_LENGTH_NOT_GRASPED = 80
HISTORY_LENGTH_GRASPED = 60

# K target a seconda dell'azione (puoi adattare)
# K_DOWN = 50.0
# K_UP   = 70.0
K_DOWN = 30.0
K_UP   = 30.0

# Abilita/disabilita pipeline YOLO (False => modalità manuale)
YOLO_ENABLED = False

# =======================
# Script
# =======================
def main():
    # --- Stato per isteresi YOLO ---
    history_not_gr = deque(maxlen=HISTORY_LENGTH_NOT_GRASPED)
    history_gr     = deque(maxlen=HISTORY_LENGTH_GRASPED)
    state = 'stand_no_box'    # 'stand_no_box' -> 'bend_no_box' -> 'stand_with_box'
    last_action = None

    # --- YOLO/RealSense opzionali ---
    if YOLO_ENABLED:
        model = YOLO(YOLO_MODEL_PATH)
        assert model, "Failed to load YOLO model"

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)

    # --- ROS 2 init ---
    rclpy.init()
    node = rclpy.create_node('vision_command_node')

    # Service client
    traj_client = node.create_client(RunTrajectory, 'run_trajectory')
    adm_client  = node.create_client(SetAdmittanceParams, 'set_admittance_params')

    have_traj = traj_client.wait_for_service(timeout_sec=1.0)
    have_adm  = adm_client.wait_for_service(timeout_sec=1.0)
    if not (have_traj and have_adm):
        node.get_logger().warn("ROS services not available yet. Actions will be skipped until services appear.")

    # Publisher per il gate
    box_gate_pub = node.create_publisher(Bool, 'perception/box_gate', 10)
    box_gate = False  # latched; all'avvio FALSE

    # UI disponibile?
    have_display = bool(os.environ.get("DISPLAY"))
    if have_display and YOLO_ENABLED:
        cv2.namedWindow("YOLO RealSense Stream", cv2.WINDOW_NORMAL)

    # --- Helper ROS2 ---
    def send_trajectory_request(direction: str):
        if not traj_client.service_is_ready():
            node.get_logger().warn("run_trajectory not ready; skipping command.")
            return
        req = RunTrajectory.Request()
        req.trajectory_type = direction
        fut = traj_client.call_async(req)
        rclpy.spin_until_future_complete(node, fut)

    def set_admittance_params(k_val: float):
        if not adm_client.service_is_ready():
            node.get_logger().warn("set_admittance_params not ready; skipping.")
            return
        req = SetAdmittanceParams.Request()
        req.k = k_val
        fut = adm_client.call_async(req)
        rclpy.spin_until_future_complete(node, fut)

    def publish_box_gate(new_val: bool):
        nonlocal box_gate
        if new_val != box_gate:
            box_gate = new_val
            box_gate_pub.publish(Bool(data=box_gate))
            node.get_logger().info(f"[BOX_GATE] → {box_gate}")

    print("Starting control loop... Press 'q' in the window or Ctrl+C in terminal to exit.")
    try:
        while rclpy.ok():
            action = None
            action_source = None  # "vision" or "manual"

            if YOLO_ENABLED:
                # --- Acquire frame ---
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame = np.asanyarray(color_frame.get_data())

                # --- YOLO predict ---
                results = model.predict(source=frame, verbose=False)
                boxes = results[0].boxes

                detected_grasped = False
                detected_not_grasped = False

                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        if conf < CONF_THRESHOLD:
                            continue
                        label = model.names[cls_id]
                        if label == 'Box-Grasped':
                            detected_grasped = True
                        elif label == 'Box-Not Grasped':
                            detected_not_grasped = True

                        # overlay (facoltativo)
                        if have_display:
                            xyxy = box.xyxy[0].cpu().numpy().astype(int)
                            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                            cv2.putText(frame, f"{label} {conf:.2f}", (xyxy[0], xyxy[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # --- Update history ---
                history_not_gr.append(1 if detected_not_grasped else 0)
                history_gr.append(1 if detected_grasped else 0)

                MIN_SAMPLES_NOT_GR = max(10, HISTORY_LENGTH_NOT_GRASPED // 4)
                MIN_SAMPLES_GR     = max(10, HISTORY_LENGTH_GRASPED // 4)

                pct_not_gr = (sum(history_not_gr) / len(history_not_gr)) if len(history_not_gr) > 0 else 0.0
                pct_gr     = (sum(history_gr)     / len(history_gr))     if len(history_gr)     > 0 else 0.0

                # --- FSM ---
                if state == 'stand_no_box':
                    if len(history_not_gr) >= MIN_SAMPLES_NOT_GR and pct_not_gr >= TRIGGER_PERCENTAGE_NOT_GRASPED:
                        action = 'down'
                        action_source = 'vision'
                        state = 'bend_no_box'
                        history_gr.clear()
                elif state == 'bend_no_box':
                    if len(history_gr) >= MIN_SAMPLES_GR and pct_gr >= TRIGGER_PERCENTAGE_GRASPED:
                        action = 'up'
                        action_source = 'vision'
                        state = 'stand_with_box'
                        history_not_gr.clear()

                # --- overlay debug ---
                if have_display:
                    cv2.putText(frame, f"STATE: {state}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    cv2.putText(frame, f"not_gr={pct_not_gr:.2f} ({len(history_not_gr)})", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                    cv2.putText(frame, f"grasped={pct_gr:.2f} ({len(history_gr)})", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                    if action:
                        cv2.putText(frame, f"ACTION: {action}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                    cv2.imshow("YOLO RealSense Stream", frame)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break

            else:
                # --- Modalità manuale ---
                user_input = input("Enter '1' for down, '2' for up, or 'q' to quit: ").strip().lower()
                if user_input == '1':
                    action = 'down'
                    action_source = 'manual'
                elif user_input == '2':
                    action = 'up'
                    action_source = 'manual'
                elif user_input == 'q':
                    break

            # --- Esecuzione azione (se nuova) ---
            if action and action != last_action:
                print(f"[INFO] Triggered action: {action} (source={action_source})")

                # # Aggiorna gate in base ALLA RICHIESTA
                # if action_source == 'vision':
                #     publish_box_gate(True)   # azione da YOLO → gate TRUE
                # else:
                #     publish_box_gate(False)  # azione manuale → gate FALSE
                
                publish_box_gate(True)
                # Set K
                set_admittance_params(K_DOWN if action == 'down' else K_UP)
                # Run trajectory
                send_trajectory_request(action)

                last_action = action

    finally:
        if YOLO_ENABLED:
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
