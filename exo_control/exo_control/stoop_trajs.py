#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from exo_interfaces.srv import RunTrajectory
import math

class LumbarTrajectoryService(Node):
    def __init__(self):
        super().__init__('lumbar_trajectory_service')

        # Parameters
        self.theta_max = math.radians(70.0)
        self.duration = 2.0
        self.rate = 50  # Hz
        self.dt = 1.0 / self.rate
        self.num_points = int(self.duration * self.rate)

        # Precompute trajectories
        self.down_traj = [self.quintic_trajectory(t * self.dt, self.duration, self.theta_max) for t in range(self.num_points)]
        self.up_traj = [self.theta_max - val for val in self.down_traj]

        # Runtime state
        self.current_traj = []
        self.traj_index = 0
        self.is_playing = False

        # ROS setup
        # self.publisher_ = self.create_publisher(Float64, 'theta_ref', 10)
        self.service_ = self.create_service(RunTrajectory, 'run_trajectory', self.handle_trajectory_request)
        self.timer_ = self.create_timer(self.dt, self.publish_next_point)

        # Publishers for trajectory STATE
        # -------------------------------
        self.theta_cmd_pub   = self.create_publisher(Float64, 'theta_ref', 10)          # COMANDO (solo in movimento)
        self.theta_state_pub = self.create_publisher(Float64, 'theta_ref_state', 10)    # STATO (sempre)

        self.theta_current = 0.0
        self.state_timer_ = self.create_timer(0.1, self.publish_hold_state)  # 10 Hz costante
        # -------------------------------


        self.get_logger().info('Trajectory service ready. Call /run_trajectory to play a motion.')

    def quintic_trajectory(self, t, T, theta_max):
        tau = t / T
        tau = min(max(tau, 0.0), 1.0)
        return theta_max * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)

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
        response.message = f"Started '{direction}' trajectory."
        self.get_logger().info(response.message)
        return response

    # def publish_next_point(self):
    #     if not self.is_playing:
    #         return

    #     if self.traj_index < len(self.current_traj):
    #         msg = Float64()
    #         msg.data = self.current_traj[self.traj_index]
    #         self.publisher_.publish(msg)
    #         self.traj_index += 1
    #     else:
    #         self.is_playing = False
    #         self.get_logger().info("Trajectory complete.")

    def publish_next_point(self):
        if not self.is_playing:
            return

        if self.traj_index < len(self.current_traj):
            theta = self.current_traj[self.traj_index]
            self.theta_current = theta  # aggiorna stato corrente

            # Pubblica SOLO il COMANDO durante la traiettoria
            self.theta_cmd_pub.publish(Float64(data=theta))

            self.traj_index += 1
        else:
            self.is_playing = False
            self.get_logger().info("Trajectory complete.")

    def publish_hold_state(self):
        # Pubblica SEMPRE lo stato corrente per il plot
        self.theta_state_pub.publish(Float64(data=self.theta_current))

def main(args=None):
    rclpy.init(args=args)
    node = LumbarTrajectoryService()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

