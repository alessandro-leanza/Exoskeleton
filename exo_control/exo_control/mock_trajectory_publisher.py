#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import math
import time

class MockTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('mock_trajectory_publisher')
        
        # This publisher sends commands directly to the motor controller nodes
        self.publisher_ = self.create_publisher(Float32MultiArray, 'position_cmd', 10)
        
        # Parameters for the sine wave trajectory
        self.declare_parameter('frequency', 0.2)  # Hz
        self.declare_parameter('amplitude', 0.4)  # Radians
        self.declare_parameter('frequency_2', 1.0) # Hz, higher frequency
        self.declare_parameter('amplitude_2', 0.1) # Radians, lower amplitude
        
        self.frequency = self.get_parameter('frequency').get_parameter_value().double_value
        self.amplitude = self.get_parameter('amplitude').get_parameter_value().double_value
        self.frequency_2 = self.get_parameter('frequency_2').get_parameter_value().double_value
        self.amplitude_2 = self.get_parameter('amplitude_2').get_parameter_value().double_value
        
        # Create a timer to publish at 50 Hz
        timer_period = 0.02  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.start_time = time.time()
        self.get_logger().info(f"Publishing sine wave trajectory to '/position_cmd'.")
        self.get_logger().info(f"Primary wave:   Amplitude={self.amplitude} rad, Frequency={self.frequency} Hz")
        self.get_logger().info(f"Secondary wave: Amplitude={self.amplitude_2} rad, Frequency={self.frequency_2} Hz")

    def timer_callback(self):
        elapsed_time = time.time() - self.start_time
        
        # Calculate the value for the primary (slower) sine wave
        pos_value_1 = self.amplitude * math.sin(2 * math.pi * self.frequency * elapsed_time)
        
        # Calculate the value for the secondary (faster) sine wave
        pos_value_2 = self.amplitude_2 * math.sin(2 * math.pi * self.frequency_2 * elapsed_time)
        
        # The final position is the sum of both waves
        pos_value = pos_value_1 + pos_value_2
        
        # Create the message, assuming two motors with opposite motion
        msg = Float32MultiArray()
        msg.data = [pos_value, -pos_value]
        
        self.publisher_.publish(msg)
        self.get_logger().debug(f'Publishing: [{pos_value:.3f}, {-pos_value:.3f}]')

def main(args=None):
    rclpy.init(args=args)
    node = MockTrajectoryPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()