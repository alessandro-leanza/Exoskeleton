#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from exo_interfaces.srv import RunTrajectory, SetAdmittanceParams

class TrajYoloClient(Node):
    def __init__(self):
        super().__init__('traj_yolo_client')

        self.run_traj_client = self.create_client(RunTrajectory, 'run_trajectory')
        self.set_adm_client = self.create_client(SetAdmittanceParams, 'set_admittance_params')

        self.run_traj_client.wait_for_service()
        self.set_adm_client.wait_for_service()

        self.subscription = self.create_subscription(String, 'yolo_trajectory_cmd', self.cmd_callback, 10)
        self.get_logger().info("Trajectory Yolo client initialized and listening.")

        # Optional: default admittance stiffness
        self.k_down = 30.0
        self.k_up = 80.0

    def cmd_callback(self, msg: String):
        cmd = msg.data.lower()
        if cmd == "down":
            self.call_set_admittance_params(self.k_down)
            self.call_run_trajectory("down")
        elif cmd == "up":
            self.call_set_admittance_params(self.k_up)
            self.call_run_trajectory("up")
        else:
            self.get_logger().warn(f"Unknown command: {cmd}")

    def call_run_trajectory(self, direction):
        req = RunTrajectory.Request()
        req.trajectory_type = direction
        future = self.run_traj_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result():
            self.get_logger().info(f"Started trajectory: {direction}")

    def call_set_admittance_params(self, k):
        req = SetAdmittanceParams.Request()
        req.k = k
        future = self.set_adm_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result():
            self.get_logger().info(f"Set admittance K = {k}")

def main(args=None):
    rclpy.init(args=args)
    node = TrajYoloClient()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
