#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

import asyncio
import moteus
import math
import signal
import sys
import threading

class MoteusDualControllerNode(Node):
    def __init__(self):
        super().__init__('moteus_dual_controller_node')

        # Declare parameters
        self.declare_parameter('motor_id_1', 1)
        self.declare_parameter('motor_id_2', 2)
        self.declare_parameter('gear_ratio', 30.0)
        self.declare_parameter('kp_scale', 1.0)
        self.declare_parameter('kd_scale', 1.0)

        self.motor_id_1 = self.get_parameter('motor_id_1').get_parameter_value().integer_value
        self.motor_id_2 = self.get_parameter('motor_id_2').get_parameter_value().integer_value
        self.gear_ratio = self.get_parameter('gear_ratio').get_parameter_value().double_value
        self.kp_scale = self.get_parameter('kp_scale').get_parameter_value().double_value
        self.kd_scale = self.get_parameter('kd_scale').get_parameter_value().double_value
        
        self.get_logger().info(f"Moteus IDs: {self.motor_id_1}, {self.motor_id_2} | Gear Ratio: {self.gear_ratio}")

        # ROS Publishers and Subscribers
        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.position_sub = self.create_subscription(Float32MultiArray, 'position_cmd', self.position_callback, 10)

        self.joint_name_1 = f'moteus_joint_{self.motor_id_1}'
        self.joint_name_2 = f'moteus_joint_{self.motor_id_2}'
        
        self.command_pos_rad = 0.0
        self.is_running = True

        # Custom Query Resolution
        self.qr = moteus.QueryResolution()
        self.qr._extra = {
            moteus.Register.POSITION: moteus.F32,
            moteus.Register.VELOCITY: moteus.F32,
            moteus.Register.TORQUE: moteus.F32,
            moteus.Register.FAULT: moteus.F32,
        }

        # --- ASYNCIO THREAD SETUP ---
        self.moteus_loop = asyncio.new_event_loop()
        self.moteus_thread = threading.Thread(target=self.run_moteus_loop, daemon=True)
        self.moteus_thread.start()

    def run_moteus_loop(self):
        asyncio.set_event_loop(self.moteus_loop)
        try:
            self.moteus_loop.run_until_complete(self.moteus_control_loop())
        except Exception as e:
            self.get_logger().error(f"Asyncio loop crash: {e}")

    def position_callback(self, msg: Float32MultiArray):
        if len(msg.data) > 0:
            self.command_pos_rad = msg.data[0]

    def publish_joint_states(self, results):
        if not results or len(results) < 2:
            return
            
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [self.joint_name_1, self.joint_name_2]

        positions, velocities, efforts = [], [], []

        for res in results:
            # Motor rotations to Output Radians
            pos = res.values[moteus.Register.POSITION] * 2 * math.pi / self.gear_ratio
            vel = res.values[moteus.Register.VELOCITY] * 2 * math.pi / self.gear_ratio
            trq = res.values[moteus.Register.TORQUE] * self.gear_ratio
            
            positions.append(pos)
            velocities.append(vel)
            efforts.append(trq)

        msg.position = positions
        msg.velocity = velocities
        msg.effort = efforts
        self.joint_state_pub.publish(msg)

    async def moteus_control_loop(self):
        self.get_logger().info("Connecting to CAN transport...")
        
        # Initialize transport inside the async thread
        transport = moteus.get_singleton_transport()
        
        c1 = moteus.Controller(id=self.motor_id_1, transport=transport)
        c2 = moteus.Controller(id=self.motor_id_2, transport=transport)

        # Ensure motors are stopped/idled before starting
        await transport.cycle([c1.make_stop(), c2.make_stop()])

        self.get_logger().info("Moteus loop active.")

        while self.is_running and rclpy.ok():
            # Calculate motor rotations (revs)
            # Motor 1: Normal | Motor 2: Inverted
            revs1 = (self.command_pos_rad / (2 * math.pi)) * self.gear_ratio
            revs2 = -(self.command_pos_rad / (2 * math.pi)) * self.gear_ratio

            # Create a combined CAN frame for efficiency
            commands = [
                c1.make_position(
                    position=revs1, query=self.qr, 
                    kp_scale=self.kp_scale, kd_scale=self.kd_scale,
                    accel_limit=100.0, velocity_limit=50.0
                ),
                c2.make_position(
                    position=revs2, query=self.qr, 
                    kp_scale=self.kp_scale, kd_scale=self.kd_scale,
                    accel_limit=100.0, velocity_limit=50.0
                )
            ]

            try:
                # Send and Receive in one burst
                results = await transport.cycle(commands)
                
                if results:
                    self.publish_joint_states(results)
                    
                    # Check for faults
                    for i, res in enumerate(results):
                        fault = res.values[moteus.Register.FAULT]
                        if fault != 0:
                            self.get_logger().error(f"Motor {commands[i].id} Fault: {fault}")

            except Exception as e:
                self.get_logger().warn(f"CAN Communication error: {e}")

            await asyncio.sleep(0.01) # ~100Hz

        # Final safety stop
        await transport.cycle([c1.make_stop(), c2.make_stop()])

def main(args=None):
    rclpy.init(args=args)
    node = MoteusDualControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.is_running = False
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()