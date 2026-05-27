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
 
class MoteusControllerNode(Node):
    def __init__(self):
        super().__init__('moteus_controller_node')
 
        # Declare parameters for motor ID and gear ratio
        self.declare_parameter('motor_id', 2)
        self.declare_parameter('gear_ratio', 1.0)
        # self.declare_parameter('gear_ratio', 1.0)
        self.declare_parameter('kp_scale', 1.0)
        self.declare_parameter('kd_scale', 1.0)
        self.declare_parameter('control_window_deg', 25.0)
 
        self.motor_id = self.get_parameter('motor_id').get_parameter_value().integer_value
        self.gear_ratio = self.get_parameter('gear_ratio').get_parameter_value().double_value
        self.kp_scale = self.get_parameter('kp_scale').get_parameter_value().double_value
        self.kd_scale = self.get_parameter('kd_scale').get_parameter_value().double_value
        self.control_window_deg = self.get_parameter('control_window_deg').get_parameter_value().double_value
        self.control_window_rad = math.radians(self.control_window_deg)
        
        self.get_logger().info(f"Initializing Moteus controller for motor ID: {self.motor_id}")
        self.get_logger().info(f"Using Kp_scale: {self.kp_scale}, Kd_scale: {self.kd_scale}")
        self.get_logger().info(f"Control active within +/- {self.control_window_deg:.2f} deg window")
 
        # ROS Publishers and Subscribers
        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.position_sub = self.create_subscription(Float32MultiArray, 'position_cmd', self.position_callback, 10)
 
        # Member variables for state and commands
        self.joint_name = f'moteus_joint_{self.motor_id}'
        self.command_pos_rad = 0.0
        self.last_state = None
        self.is_running = True
        self.in_hold = False
 
        self.qr = moteus.QueryResolution()
        self.qr._extra = {
            moteus.Register.CONTROL_POSITION : moteus.F32,
            moteus.Register.CONTROL_VELOCITY : moteus.F32,
            moteus.Register.CONTROL_TORQUE   : moteus.F32,
            moteus.Register.POSITION_ERROR   : moteus.F32,
            moteus.Register.VELOCITY_ERROR   : moteus.F32,
            moteus.Register.TORQUE_ERROR     : moteus.F32,
            moteus.Register.POSITION         : moteus.F32,  # <-- we need this to read initial pos
            moteus.Register.FAULT            : moteus.F32, # <-- Add this to see fault codes
        }
 
        # Set up graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown_handler)
 
        # --- CORRECTED ASYNCIO/THREADING SETUP ---
        # Create a new asyncio event loop for the background thread
        self.moteus_loop = asyncio.new_event_loop()
        # Run the asyncio event loop in a separate thread
        self.moteus_thread = threading.Thread(target=self.run_moteus_loop, daemon=True)
        self.moteus_thread.start()
 
        # Log position/velocity/torque every second
        self.state_log_timer = self.create_timer(1.0, self.log_state)
 
    def run_moteus_loop(self):
        """Runs the asyncio event loop and the main control coroutine."""
        asyncio.set_event_loop(self.moteus_loop)
        self.moteus_loop.run_until_complete(self.moteus_control_loop())
 
    def position_callback(self, msg: Float32MultiArray):
 
        """Receives position commands. For single motor mode, it uses the first value."""
        if len(msg.data) > 0:
            self.command_pos_rad = msg.data[0]
        else:
            self.get_logger().warn(f"Received empty position_cmd message.")
 
    def publish_joint_state(self, state):
        """Publishes the current state of the motor to the /joint_states topic."""
        if not state:
            return
            
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        # To satisfy the admittance controller, we "fake" a second joint.
        msg.name = [self.joint_name, f'{self.joint_name}_mirrored']
 
        # Convert from motor rotations to radians at the output
        position_rad, velocity_rad_s, torque_nm = self.convert_state(state)
 
        # The admittance controller expects two joints, where one is the negative of the other.
        # We publish the real state and a mirrored virtual state.
        msg.position = [position_rad, -position_rad]
        msg.velocity = [velocity_rad_s, -velocity_rad_s]
        msg.effort = [torque_nm, -torque_nm]
        
        self.joint_state_pub.publish(msg)
 
    def convert_state(self, state):
        position_rad = state.values[moteus.Register.POSITION] * 2 * math.pi / self.gear_ratio
        velocity_rad_s = state.values[moteus.Register.VELOCITY] * 2 * math.pi / self.gear_ratio
        torque_nm = state.values[moteus.Register.TORQUE] * self.gear_ratio
        return position_rad, velocity_rad_s, torque_nm
 
    def log_state(self):
        """Logs position/velocity/torque once per second."""
        state = self.last_state
        if not state:
            return
 
        position_rad, velocity_rad_s, torque_nm = self.convert_state(state)
        self.get_logger().info(
            f"State: pos={position_rad:.3f} rad, vel={velocity_rad_s:.3f} rad/s, torque={torque_nm:.3f} Nm"
        )
 
    def is_within_control_window(self):
        if not self.last_state:
            return False
        position_rad, _, _ = self.convert_state(self.last_state)
        return abs(position_rad - self.command_pos_rad) <= self.control_window_rad
 
    async def moteus_control_loop(self):
        """The main asyncio loop for communicating with the moteus controller."""
        self.get_logger().info("Starting Moteus control loop...")
        
        # Create a controller instance
        c = moteus.Controller(id=self.motor_id)
        
        try:
            # Stop the controller to ensure it's in a known state
            await c.set_stop()
 
            # *** KEY CHANGE: Read initial position before starting the loop ***
            # This ensures our first command is relative to the actual starting position.
            initial_state = await c.query(query_override=self.qr)
            if initial_state:
                initial_pos_rev = initial_state.values[moteus.Register.POSITION]
                self.command_pos_rad = initial_pos_rev * 2 * math.pi / self.gear_ratio
                self.get_logger().info(f"Initialized command position to motor's start: {self.command_pos_rad:.3f} rad")
            else:
                self.get_logger().warn("Could not read initial motor position. Starting at 0.0 rad.")
 
            while self.is_running and rclpy.ok():
                # Convert commanded position from radians to motor rotations
                cmd_pos_motor_rev = self.command_pos_rad / (2 * math.pi) * self.gear_ratio
 
                if self.is_within_control_window():
                    # Send the position command and query for the latest state.
                    # This is an efficient way to both command and get feedback in one CAN frame.
                    state = await c.set_position(
                        position=cmd_pos_motor_rev,
                        query_override=self.qr,
                        accel_limit=100.0,
                        velocity_limit=50.0,
                        kp_scale=self.kp_scale,
                        kd_scale=self.kd_scale,
                    )
                    self.in_hold = True
                else:
                    if self.in_hold:
                        await c.set_stop()
                        self.in_hold = False
                    state = await c.query(query_override=self.qr)
 
                if state:
                    self.last_state = state
                    self.publish_joint_state(state)
                    fault_code = state.values[moteus.Register.FAULT]
                    if fault_code != 0:
                        self.get_logger().error(f"Moteus motor {self.motor_id} reported fault: {fault_code}")
 
                else:
                    self.get_logger().warn("No response from Moteus controller.")
 
                # Slowing the loop from 1000Hz to 250Hz can improve stability and reduce
                # vibration caused by noisy velocity estimates.
                await asyncio.sleep(0.008)
 
        except Exception as e:
            self.get_logger().error(f"An error occurred in the Moteus control loop: {e}")
        finally:
            self.get_logger().info("Moteus control loop stopping. Setting motor to idle.")
            await c.set_stop()
 
    def shutdown_handler(self, signum, frame):
        """Handles Ctrl+C to gracefully shut down the node and motor."""
        self.get_logger().info("Ctrl+C detected! Shutting down...")
        self.is_running = False
        # The asyncio loop will see is_running=False and exit, stopping the motor.
        # Allow some time for the loop to stop.
        rclpy.shutdown()
        sys.exit(0)
 
def main(args=None):
    rclpy.init(args=args)
    node = MoteusControllerNode()
    try:
        # rclpy.spin() will run the node and process callbacks until shutdown is called.
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down.")
    except Exception as e:
        node.get_logger().fatal(f"Unhandled exception in main: {e}")
    finally:
        # The shutdown_handler is responsible for stopping the motor and calling rclpy.shutdown()
        # which will cause spin() to exit. We just need to destroy the node here.
        node.is_running = False # Ensure the asyncio loop stops
        node.destroy_node()
 
        # The shutdown_handler will be called on Ctrl+C, which calls rclpy.shutdown()
        # This will cause rclpy.spin() to exit.
        # We ensure shutdown is called regardless.
        if rclpy.ok():
            rclpy.shutdown()
 
if __name__ == '__main__':
    main()
