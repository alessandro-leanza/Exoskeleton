# Laboratory setup and operation

[Back to the project overview](../README.md)

These instructions describe the existing laboratory workspace. They are not a
complete installation recipe for a new machine: Python/ML dependencies are not
pinned, and the ROS package metadata does not declare the full runtime stack.
For perception experiments without hardware, start with the
[offline benchmark guide](OFFLINE_BENCHMARK.md).

## Prerequisites

| Area | Existing setup |
| --- | --- |
| ROS | ROS 2 Humble, sourced from `/opt/ros/humble/setup.bash`; `colcon` for building |
| Python | Python 3.10; a virtual environment at `~/exo_v2_ws/ros2_env` |
| Workspace | Repository at `~/exo_v2_ws/src/Exoskeleton` |
| Actuation | Two configured Moteus controllers and a compatible CAN interface |
| Wearable perception | Tobii Glasses 3 reachable over Wi-Fi; `g3pylib` and the Kivy application dependencies |
| ML | PyTorch, Transformers with Qwen2.5-VL support, Accelerate, Ultralytics, OpenCV, NumPy, and Pillow; a suitable GPU is recommended for local VLM inference |

The environment is activated with `source ros2_env/bin/activate`; it is a Python
virtual environment, despite older notes referring to it as Conda. If Conda is
active, deactivate it before sourcing ROS and the virtual environment.

## Build the workspace

From the existing workspace:

```bash
cd ~/exo_v2_ws
source /opt/ros/humble/setup.bash
source ros2_env/bin/activate
colcon build --symlink-install --base-paths src/Exoskeleton
```

In **each terminal** used below, source the environment and workspace overlay:

```bash
cd ~/exo_v2_ws
source /opt/ros/humble/setup.bash
source ros2_env/bin/activate
source install/setup.bash
```

## Configure perception and hardware

- In [`tobii_vlm/__main__.py`](../exo_control/exo_control/tobii_vlm/__main__.py),
  set `YOLO_MODEL_PATH` to your trained model. The current default is an absolute
  laboratory path ending in `best-tobii-3objs-v2.pt`; **that file is not tracked
  in this repository**. Other detector checkpoints are present, but their labels
  and behavior must be checked before substitution.
- The live app initializes `Qwen/Qwen2.5-VL-7B-Instruct` through Hugging Face.
  Ensure the model is available and the machine has sufficient memory.
- The optional remote weight-estimation request runs only when `OPENAI_API_KEY`
  is set and sends a scene image to the API. The local Qwen path does not require
  this key. Neither estimate automatically updates the controller's payload mass.
- Check the motor IDs, gear ratio, direction conventions, and device access for
  your physical setup. Current Moteus configuration notes are in
  [`exo_config/moteus/readme.md`](../exo_config/moteus/readme.md).
- Review the assistance model and limits in
  [`admittancecontrol_box.py`](../exo_control/exo_control/admittancecontrol_box.py),
  including `m_w`, `m_b`, lever arms, `coeff_assist`, and torque slew rates.
  The controller initializes its internal `box_gate` to `True`; do not assume
  assistance waits for the first perception message.

These commands operate physical actuators. Use the calibrated laboratory setup
and verify the configured limits and stop procedure before enabling the drives.

## Run the stack

The following are the existing launch commands. Use separate terminals with the
environment sourced as above; startup behavior must be checked for your setup.

### Moteus motor interface

```bash
cd ~/exo_v2_ws/src/Exoskeleton/exo_control/exo_control
python moteus_control_node.py
```

Connects to both controllers, publishes `joint_states`, subscribes to
`position_cmd`, reports motor faults, and requests idle on shutdown.

### Admittance controller

```bash
ros2 run exo_control admittancecontrol_box
```

Reads joint feedback and perception cues, runs the pick/place finite-state
machine, and publishes assistive commands and telemetry. Runtime admittance
tuning is exposed through the `set_admittance_params` service.

### Tobii application

```bash
cd ~/exo_v2_ws/src/Exoskeleton/exo_control/exo_control
python tobii_vlm
```

Connect the glasses over Wi-Fi, select them in the application, then choose
**Live → Start**. The detector weights are specific to the laboratory setting.
See [implementation scope](RESEARCH.md#implementation-scope) for the distinction
between the perception gate and the experimental VLM outputs.

### Telemetry

If PlotJuggler is installed:

```bash
ros2 run plotjuggler plotjuggler
```

Useful topics include `joint_states`, `perception/box_gate`,
`admittance/fsm_state_id`, `admittance/assist_active`, and
`admittance/tau_ass_total`.

## Legacy ODrive hardware

The scripts below belong to the **previous ODrive motor setup** and are not a
Moteus calibration procedure. They are retained here from the original README.

1. Verify the encoders:

   ```bash
   cd ~/exo_v2_ws/src/Exoskeleton/test_scripts
   python read2encoders.py
   ```

2. For that legacy setup, manually turn the right motor very low; the left can
   be slightly higher. The right motor is used for synchronization. If the left
   is too low, it may hit the screw during the downward calibration move.
3. Start `odrivetool`, then run the calibration sequence:

   ```python
   odrv0.axis0.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
   odrv1.axis0.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
   ```

   In the original setup, both motors beep and move forward/backward, depending
   on mounting direction. If that does not happen, stop and check the cabling.
