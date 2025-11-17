#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState

import odrive
from odrive.enums import CONTROL_MODE_TORQUE_CONTROL, CONTROL_MODE_POSITION_CONTROL
from odrive.enums import INPUT_MODE_PASSTHROUGH
from odrive.enums import AXIS_STATE_CLOSED_LOOP_CONTROL, AXIS_STATE_IDLE

import math
import signal
import sys

from exo_interfaces.srv import SetControlMode

import time

MODE_NONE = 0 
MODE_POSITION = CONTROL_MODE_POSITION_CONTROL
MODE_TORQUE = CONTROL_MODE_TORQUE_CONTROL

SERIAL_NUMBER_0 = "317532613431"
SERIAL_NUMBER_1 = "365A388C3131"


class ODriveControllerNode(Node):
    def __init__(self):
        super().__init__('odrive_dual_controller_node')

        # Fill in actual serial numbers here
        self.get_logger().info("Connecting to ODrive 0")
        self.odrive0 = odrive.find_any(serial_number= SERIAL_NUMBER_0)
        self.get_logger().info("Connecting to ODrive 1")
        self.odrive1 = odrive.find_any(serial_number= SERIAL_NUMBER_1)

        self.axis_r = self.odrive0.axis0
        self.axis_l = self.odrive1.axis0
        self.get_logger().info("Both ODrives connected.")

        self.joint_names = ['joint_0', 'joint_1']

        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.torque_sub = self.create_subscription(Float32MultiArray, 'torque_cmd', self.torque_callback, 10)
        self.position_sub = self.create_subscription(Float32MultiArray, 'position_cmd', self.position_callback, 10)
        self.mode_srv = self.create_service(SetControlMode, 'set_control_mode', self.set_control_mode)

        self.odrive0.clear_errors()
        self.odrive1.clear_errors()

        for axis, i in zip([self.axis_r, self.axis_l], [0, 1]):
            if not axis.motor.is_calibrated:
                self.get_logger().error(f"Axis {i} not calibrated.")
                return

        self.current_mode = MODE_POSITION
        # self.current_mode = MODE_NONE
        
        self.set_odrive_mode(self.current_mode)

        self.torque_0 = 0.0
        self.torque_1 = 0.0
        self.pos_0 = 0.0
        self.pos_1 = 0.0

        self.GEAR_RATIO = 30.0
        self.positions_initialized = False

        self.create_timer(0.005, self.publish_joint_states)
        self.create_timer(1.0, self.monitor_errors)
        self.create_timer(0.005, self.send_commands)

        signal.signal(signal.SIGINT, self.shutdown_handler)

    def set_odrive_mode(self, mode_int):
        if mode_int not in [MODE_TORQUE, MODE_POSITION, MODE_NONE]:
            self.get_logger().error(f"Invalid control mode int: {mode_int}")
            return False

        self.current_mode = mode_int

        for axis in [self.axis_r, self.axis_l]:
            if mode_int == MODE_NONE:
                axis.requested_state = AXIS_STATE_IDLE
            else:
                axis.controller.config.control_mode = mode_int
                axis.controller.config.input_mode = INPUT_MODE_PASSTHROUGH
                axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL

        mode_str = {MODE_NONE: "NONE", MODE_TORQUE: "TORQUE", MODE_POSITION: "POSITION"}[mode_int]
        self.get_logger().info(f"Set control mode to {mode_str}")
        return True

    def set_control_mode(self, request, response):
        success = self.set_odrive_mode(request.mode)
        response.success = success
        response.message = f"Control mode set to {request.mode}" if success else "Failed to set control mode"
        return response

    def torque_callback(self, msg):
        if len(msg.data) != 2:
            self.get_logger().warn("Expected 2 torque values")
            return
        self.torque_0, self.torque_1 = msg.data

    def position_callback(self, msg):
        if len(msg.data) != 2:
            self.get_logger().warn("Expected 2 position values")
            return
        self.pos_0, self.pos_1 = msg.data

    def send_commands(self):
        if self.current_mode == MODE_NONE:
            return
        elif self.current_mode == MODE_TORQUE:
            self.axis_r.controller.input_torque = self.torque_0
            self.axis_l.controller.input_torque = self.torque_1
        elif self.current_mode == MODE_POSITION:
            if not self.positions_initialized:
                self.pos_0 = self.axis_r.encoder.pos_estimate * 2 * math.pi / self.GEAR_RATIO
                self.pos_1 = self.axis_l.encoder.pos_estimate * 2 * math.pi / self.GEAR_RATIO
                self.positions_initialized = True
            self.axis_r.controller.input_pos = self.pos_0 / (2 * math.pi) * self.GEAR_RATIO
            self.axis_l.controller.input_pos = self.pos_1 / (2 * math.pi) * self.GEAR_RATIO

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [
            self.axis_r.encoder.pos_estimate * 2 * math.pi / self.GEAR_RATIO,
            self.axis_l.encoder.pos_estimate * 2 * math.pi / self.GEAR_RATIO
        ]
        msg.velocity = [
            self.axis_r.encoder.vel_estimate * 2 * math.pi / self.GEAR_RATIO,
            self.axis_l.encoder.vel_estimate * 2 * math.pi / self.GEAR_RATIO
        ]
        msg.effort = [
            self.axis_r.motor.current_control.Iq_measured * 0.0250606 * self.GEAR_RATIO,
            self.axis_l.motor.current_control.Iq_measured * 0.0250606 * self.GEAR_RATIO
        ]
        self.joint_state_pub.publish(msg)

    def monitor_errors(self):
        self.get_logger().info("Battery voltages:")
        self.get_logger().info(f"ODrive0: {self.odrive0.vbus_voltage:.2f} V")
        self.get_logger().info(f"ODrive1: {self.odrive1.vbus_voltage:.2f} V")
        for i, axis in enumerate([self.axis_r, self.axis_l]):
            if axis.motor.error or axis.encoder.error or axis.controller.error:
                self.get_logger().error(
                    f"❌ Axis {i} errors detected:\n"
                    f"   Motor error:      {axis.motor.error}\n"
                    f"   Encoder error:    {axis.encoder.error}\n"
                    f"   Controller error: {axis.controller.error}"
                )

                self.get_logger().info(f"🔧 Attempting to reset Axis {i}...")
                try:
                    if i == 0:
                        self.odrive0.clear_errors()
                    else:
                        self.odrive1.clear_errors()
                    time.sleep(0.1)
                    if self.current_mode != MODE_NONE:
                        axis.controller.config.control_mode = self.current_mode
                        axis.controller.config.input_mode = INPUT_MODE_PASSTHROUGH
                        axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
                        self.get_logger().info(f"✅ Axis {i} restarted into CLOSED_LOOP_CONTROL.")
                    else:
                        axis.requested_state = AXIS_STATE_IDLE
                        self.get_logger().info(f"Axis {i} set to IDLE (MODE_NONE).")
                except Exception as e:
                    self.get_logger().error(f"⚠️ Failed to reset Axis {i}: {str(e)}")


    def shutdown_handler(self, signum, frame):
        self.get_logger().info("Ctrl+C detected! Stopping ODrives...")
        for i, axis in enumerate([self.axis_r, self.axis_l]):
            axis.requested_state = AXIS_STATE_IDLE
            self.get_logger().info(f"Axis {i} set to IDLE.")
        rclpy.shutdown()
        sys.exit(0)

def main(args=None):
    rclpy.init(args=args)
    node = ODriveControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.shutdown_handler(None, None)
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()