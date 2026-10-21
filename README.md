# Context-Aware Exoskeleton Assistance

A research framework connecting egocentric vision and gaze tracking with
variable-admittance control for an active back-support exoskeleton. Task-context
cues complement the user's posture to inform **when assistance starts** and
**how much support is delivered** during manual handling.

[Project overview](https://alessandro-leanza.github.io/projects/exoskeleton/) ·
[Earlier preprint (arXiv)](https://arxiv.org/abs/2602.04648)

## Perception–control architecture

<p align="center">
  <a href="images/architecture.png">
    <img src="images/architecture.png" alt="High-level perception provides visual grasp-state and payload cues to a finite-state machine and variable-admittance controller, which drives the back-support exoskeleton" width="1000">
  </a>
</p>

- **Attend:** egocentric vision and gaze associate the user's attention with the
  objects visible through the Tobii glasses.
- **Interpret:** visual grasp-state cues inform task progression; asynchronous
  vision-language inference estimates payload information in the experimental
  framework without blocking the time-critical assistance gate.
- **Assist:** a finite-state policy combines perception and posture with
  variable-admittance control, adapting assistance and interaction parameters.

## Selected experimental results

The evaluations below concern controlled laboratory stooping and lifting tasks.
The linked arXiv preprint is an earlier version and may not contain all figures
or experimental results summarized here.

| Evaluation | Selected finding | Sample and comparison |
| --- | --- | --- |
| Gaze ablation | Mean payload-aware assistance-onset delay: **0.066 ± 0.033 s** with gaze versus **1.892 ± 0.040 s** without gaze; mean anticipation **1.826 s** | One participant, 50 randomized trials per condition. Onset is referenced to the first YOLO detection of the object as grasped, not a direct measurement of physical load transfer. |
| Questionnaire | Physical-demand Q1 score: **3.60 / 4** with vision gating, **2.13 / 4** with posture-only assistance, **2.00 / 4** without the exoskeleton. Higher means less perceived physical demand. | 15 participants; five repetitions per condition. These ratings evaluate the integrated assistance strategy. |
| Preliminary EMG | Loaded-condition envelope RMS reduction: **14.29%** for thoracic and **14.63%** for lumbar erector spinae | Five participants; complete vision-enabled exoskeleton versus no exoskeleton during loaded lifting. |

The preliminary EMG study did not include a posture-only assistance condition,
so these muscular-activation reductions cannot be independently attributed to
gaze or vision gating. The current experiments also do not independently separate
task-phase anticipation from payload-dependent torque adaptation. These findings
do not establish performance across arbitrary users, objects, or environments.

## Experimental setup

<p align="center">
  <a href="images/setup.webp">
    <img src="images/setup.webp" alt="Participant lifting a box while wearing the active back-support exoskeleton and Tobii glasses; an inset shows the egocentric view and gaze point" width="340">
  </a>
</p>

*Back-support exoskeleton during the lifting task. Click either figure to view it
at full resolution. See [images/README.md](images/README.md) for figure provenance.*

## Control stack and setup

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
