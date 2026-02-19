# Exoskeleton 

This repository contains the ROS 2 / Moteus / Tobii-based control stack for a back-support exoskeleton.

The typical workflow is:
1. Set up the ROS 2 + Conda environment.
2. Run:
   - the Moteus motors controller node,
   - the admittance controller node,
   - the Tobii glasses application.
3. Calibrate the ODrive axes. (OLD MOTORS)

Below is a step-by-step guide.

---

## Environment setup

The repository assumes:
- ROS 2 Humble installed (sourced from `/opt/ros/humble`).
- A Conda environment called `ros2_env` in `~/exo_v2_ws`.
- This repository located at: `~/exo_v2_ws/src/Exoskeleton`.

For convenience, you can define two aliases (e.g. in your `~/.bashrc`):
- Build: `ec`
  ```bash
  alias ec='conda deactivate >/dev/null 2>&1; cd ~/exo_v2_ws && source /opt/ros/humble/setup.bash && source ros2_env/bin/activate && colcon build --symlink-install --base-paths src/Exoskeleton'
  ```
- Source overlay: `es`
  ```bash
  alias es='conda deactivate >/dev/null 2>&1; cd ~/exo_v2_ws && source /opt/ros/humble/setup.bash && source ros2_env/bin/activate && source install/setup.bash'
  ```

## Run stack
1) Start the moteus controller (Python):
   ```bash
   cd src/Exoskeleton/exo_control/exo_control
   python moteus_control_node.py
   ```
   Connects to both moteus controllers, publishes joint states, monitors faults, and idles the drives on shutdown.

2) Start admittance control:
   ```bash
   ros2 run exo_control admittancecontrol_box
   ```
   Runs a ROS 2 admittance controller with a pick/place finite-state machine: it reads joint states and intent cues (theta_ref, box_gate), blends gravity and box/load compensation into smooth assistive torques with slew limits, ramps stiffness/damping between soft/hard profiles, publishes position commands and telemetry (estimated/assist torques, admittance params, FSM state), and exposes a service to retune admittance gains at runtime.

3) Launch Tobii glasses app:
   ```bash
   cd src/Exoskeleton/exo_control/exo_control
   python tobii_vlm
   ```

## Tobii glasses app
- Connect the glasses over Wi‑Fi, launch `tobii`, select the glasses on the first screen, then choose Live → Start.
- The weights of YOLO need to be fine-tuned. At the moment, they are working well just in our laboratory.

## PlotJuggler 
To see in realtime the topics:
   ```bash
   ros2 run plotjuggler plotjuggler 
   ```


## Calibration (run once per setup) --> OLD MOTORS 
1) Verify encoders:
   ```bash
   cd src/Exoskeleton/test_scripts
   python read2encoders.py
   ```
2) Manually turn the right motor very low (left can be slightly higher). The right motor is used for synchronisation; if the left is set too low it may hit the screw during the downward calibration move.
3) Run ODrive calibration:
   ```bash
   odrivetool
   ```
   And then:
   ```bash
   odrv0.axis0.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
   odrv1.axis0.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
   ```
   You should hear a beep from both motors. After that, they must move forward and then backward (or vice versa, depending on the direction in which they are mounted). If not, recheck and tighten the cabling.
