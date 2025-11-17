#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from exo_interfaces.srv import RunTrajectory, SetAdmittanceParams


# qui decido se andare in discesa o in salita
# posso decidere i parametri di ammettenza
# NELLA PRATICO A ME INTERESSA STO CODICE, DOVRO' INSERIRE QUI I PARAMETRI DI AMMETTENZA (e la traiettoria in futuro)
class TrajectoryAdmittanceClient(Node):
    def __init__(self):
        super().__init__('trajectory_admittance_client')

        self.run_trajectory_client = self.create_client(RunTrajectory, 'run_trajectory')
        self.set_admittance_client = self.create_client(SetAdmittanceParams, 'set_admittance_params')

        # Wait for services to be available
        self.get_logger().info('Waiting for services...')
        self.run_trajectory_client.wait_for_service()
        self.set_admittance_client.wait_for_service()
        self.get_logger().info('Services available!')

    def call_run_trajectory(self, trajectory_type: str):
        req = RunTrajectory.Request()
        req.trajectory_type = trajectory_type
        future = self.run_trajectory_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info(f'RunTrajectory response: success={future.result().success}, message="{future.result().message}"')
        else:
            self.get_logger().error('RunTrajectory service call failed')

    def call_set_admittance_params(self, k):
        req = SetAdmittanceParams.Request()
        req.k = k
        future = self.set_admittance_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info(f'SetAdmittanceParams response: success={future.result().success}, message="{future.result().message}"')
        else:
            self.get_logger().error('SetAdmittanceParams service call failed')

def main(args=None):
    rclpy.init(args=args)
    client = TrajectoryAdmittanceClient()

    try:
        while True:
            print("\nSelect option:")
            print("1 - Run trajectory 'down'")
            print("2 - Run trajectory 'up'")
            print("3 - Set admittance parameters (m, c, k)")
            print("q - Quit")

            choice = input("Enter choice: ").strip()
            if choice == '1':
                client.call_run_trajectory('down')
            elif choice == '2':
                client.call_run_trajectory('up')
            elif choice == '3':
                try:
                    k_str = input("Enter k values: ")
                    k = float(k_str)

                    client.call_set_admittance_params(k)
                except ValueError:
                    print("Invalid input, please enter floats separated by space.")
            elif choice.lower() == 'q':
                print("Exiting.")
                break
            else:
                print("Invalid choice, try again.")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
