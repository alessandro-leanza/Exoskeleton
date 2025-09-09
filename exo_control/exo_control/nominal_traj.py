#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from exo_interfaces.srv import RunTrajectory
import math

class XRefTrajectoryService(Node):
    def __init__(self):
        super().__init__('xref_trajectory_service')

        # Parametri della traiettoria
        self.x_ref_max = math.radians(70.0)  # Raggiunge 70°
        self.duration = 2.0                  # secondi
        self.rate = 50                       # Hz
        self.dt = 1.0 / self.rate
        self.num_points = int(self.duration * self.rate)

        # Traiettorie precomputate
        self.down_traj = [self.quintic_trajectory(t * self.dt, self.duration, self.x_ref_max) for t in range(self.num_points)]
        self.up_traj = [self.x_ref_max - val for val in self.down_traj]

        # Stato runtime
        self.current_traj = []
        self.traj_index = 0
        self.is_playing = False

        # ROS
        self.publisher_ = self.create_publisher(Float64, 'x_ref', 10)
        self.service_ = self.create_service(RunTrajectory, 'run_trajectory', self.handle_trajectory_request)
        self.timer_ = self.create_timer(self.dt, self.publish_next_point)

        self.get_logger().info('x_ref trajectory service ready. Call /run_trajectory with "down" or "up".')

    def quintic_trajectory(self, t, T, x_max):
        tau = t / T
        tau = min(max(tau, 0.0), 1.0)
        return x_max * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)

    def handle_trajectory_request(self, request, response):
        direction = request.trajectory_type.lower()
        if direction == 'down':
            self.current_traj = self.down_traj
        elif direction == 'up':
            self.current_traj = self.up_traj
        else:
            response.success = False
            response.message = "Invalid trajectory_type. Use 'down' or 'up'."
            return response

        self.traj_index = 0
        self.is_playing = True
        response.success = True
        response.message = f"Started '{direction}' x_ref trajectory."
        self.get_logger().info(response.message)
        return response

    def publish_next_point(self):
        if not self.is_playing:
            return

        if self.traj_index < len(self.current_traj):
            msg = Float64()
            msg.data = self.current_traj[self.traj_index]
            self.publisher_.publish(msg)
            self.traj_index += 1
        else:
            self.is_playing = False
            self.get_logger().info("Trajectory complete.")

def main(args=None):
    rclpy.init(args=args)
    node = XRefTrajectoryService()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
