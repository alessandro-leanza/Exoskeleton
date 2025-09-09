# motors_control.py (nuovo)

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState

import odrive
from odrive.enums import CONTROL_MODE_POSITION_CONTROL
from odrive.enums import INPUT_MODE_PASSTHROUGH
from odrive.enums import AXIS_STATE_CLOSED_LOOP_CONTROL, AXIS_STATE_IDLE

import math
import signal
import sys
import time

from exo_interfaces.srv import SetControlMode

SERIAL_NUMBER_0 = "365A388C3131"  # Motore destro
SERIAL_NUMBER_1 = "317532613431"  # Motore sinistro365A388C3131

KT = 0.0250606  # Nm/A
GEAR_RATIO = 30.0

MODE_NONE = 0
MODE_POSITION = CONTROL_MODE_POSITION_CONTROL


class ODriveTorqueMirror(Node):
    def __init__(self):
        super().__init__('motors_control')

        self.get_logger().info("Connecting to ODrive 0 (Right motor)")
        self.odrive0 = odrive.find_any(serial_number=SERIAL_NUMBER_0)
        self.get_logger().info("Connecting to ODrive 1 (Left motor)")
        self.odrive1 = odrive.find_any(serial_number=SERIAL_NUMBER_1)

        self.axis_r = self.odrive0.axis0
        self.axis_l = self.odrive1.axis0

        self.get_logger().info("Both ODrives connected.")

        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.torque_sub = self.create_subscription(Float32MultiArray, 'torque_cmd', self.torque_callback, 10)
        self.position_sub = self.create_subscription(Float32MultiArray, 'position_cmd', self.position_callback, 10)


        self.torque = 0.0
        self.ready = False

        self.axis_r.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
        self.axis_l.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL

        for axis in [self.axis_r, self.axis_l]:
            axis.controller.config.control_mode = MODE_POSITION
            axis.controller.config.input_mode = INPUT_MODE_PASSTHROUGH

        self.create_timer(0.005, self.publish_joint_states)
        self.create_timer(0.005, self.send_position_command)


        signal.signal(signal.SIGINT, self.shutdown_handler)

    def torque_callback(self, msg):
        if len(msg.data) != 1:
            self.get_logger().warn("Expected single torque value for right motor")
            return
        self.torque = msg.data[0]
        self.ready = True

    def position_callback(self, msg):
        if len(msg.data) != 1:
            self.get_logger().warn("Expected single theta_ref value")
            return
        self.theta_ref = msg.data[0]
        self.ready = True

    def send_torque_command(self):
        if not self.ready:
            self.get_logger().warn("❌ Nessun torque_cmd ricevuto. Comandi non inviati.")
            return
        self.get_logger().info(f"[eff_cmd] τ = {self.torque:.3f} Nm → R | {-self.torque:.3f} Nm ← L")
        self.axis_r.controller.input_torque = self.torque
        self.axis_l.controller.input_torque = -self.torque

    def send_position_command(self):
        if not self.ready:
            self.get_logger().warn("❌ Nessun theta_ref ricevuto.")
            return

        pos_r = self.theta_ref
        pos_l = -self.theta_ref

        self.axis_r.controller.input_pos = pos_r / (2 * math.pi) * GEAR_RATIO
        self.axis_l.controller.input_pos = pos_l / (2 * math.pi) * GEAR_RATIO

        self.get_logger().info(f"[θ_ref] R: {pos_r:.3f} rad | L: {pos_l:.3f} rad")


    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['joint_0', 'joint_1']
        msg.position = [
            self.axis_r.encoder.pos_estimate * 2 * math.pi / GEAR_RATIO,
            self.axis_l.encoder.pos_estimate * 2 * math.pi / GEAR_RATIO
        ]
        msg.velocity = [
            self.axis_r.encoder.vel_estimate * 2 * math.pi / GEAR_RATIO,
            self.axis_l.encoder.vel_estimate * 2 * math.pi / GEAR_RATIO
        ]
        msg.effort = [
            self.axis_r.motor.current_control.Iq_measured * KT * GEAR_RATIO,
            self.axis_l.motor.current_control.Iq_measured * KT * GEAR_RATIO
        ]
        self.joint_state_pub.publish(msg)

    def shutdown_handler(self, signum, frame):
        self.get_logger().info("Ctrl+C detected! Stopping ODrives...")
        for axis in [self.axis_r, self.axis_l]:
            axis.requested_state = AXIS_STATE_IDLE
        rclpy.shutdown()
        sys.exit(0)


def main(args=None):
    rclpy.init(args=args)
    node = ODriveTorqueMirror()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.shutdown_handler(None, None)
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
