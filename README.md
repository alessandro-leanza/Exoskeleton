# Exoskeleton Setup

This repository contains the ROS 2 / ODrive / Tobii-based control stack for a back-support exoskeleton.

The typical workflow is:
1. Set up the ROS + Conda environment.
2. Calibrate the ODrive axes.
3. Run:
   - the ODrive controller node,
   - the admittance controller node,
   - the Tobii glasses application.

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

## Calibration (run once per setup)
1) Verify encoders:
   ```bash
   cd src/Exoskeleton/test_scripts
   python read2encoders.py
   ```
2) Manually turn the right motor very low (left can be slightly higher). The right motor is used for synchronisation; if the left is set too low it may hit the screw during the downward calibration move.
3) Run ODrive calibration:
   ```bash
   odrivetool
   odrv0.axis0.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
   odrv1.axis0.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
   ```
   You should hear a beep from both motors. If not, recheck and tighten the cabling.

## Run stack
1) Start ODrive bridge:
   ```bash
   ros2 run exo_control twoboards_odrive
   ```
2) Start admittance control:
   ```bash
   ros2 run exo_control admittancecontrol_box
   ```
3) Launch Tobii glasses app:
   ```bash
   cd exo_control/exo_control
   python tobii.py
   ```

## Tobii glasses app
- Connect the glasses over Wi‑Fi, launch `tobii.py`, select the glasses on the first screen, then choose Live → Start.
- YOLO labels and bounding boxes are currently commented out. If you need them, reach out and we can re-enable the relevant code.
