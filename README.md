# Exoskeleton
Build -> ec -> ec is aliased to "conda deactivate >/dev/null 2>&1; cd ~/exo_v2_ws && source /opt/ros/humble/setup.bash && source ros2_env/bin/activate && colcon build --symlink-install --base-paths src/Exoskeleton"

Source -> es -> es is aliased to "conda deactivate >/dev/null 2>&1; cd ~/exo_v2_ws && source /opt/ros/humble/setup.bash && source ros2_env/bin/activate && source install/setup.bash"

The first thing to do after setting up the environment is calibration. First, check that the encoders are working with 'cd test_scripts' 'python read2encoders.py'; then turn the right motor very low by hand (the left one can be a little higher, as the code uses the right one for synchronisation, but if you turn the left one too low during calibration, it will touch the screw because it calibrates by going down); then use 'odrivetool' and do 'odrv0.axis0.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE' and 'odrv1. axis0.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE'. You should hear a beep for both motors. If you do not hear the beep, check the cable connections and tighten the ones that move the most with a screwdriver.

Once calibration is complete, the codes to run are 1) 'ros2 run exo_control two_boards_exo_control' 2) 'ros2 run exo_control admittance_controller_perception_offset' 3) 'ros2 run exo_control stoop_trajs' (even if it does nothing for now) 4) 'cd exo_control/exo_control' 'python tobii.py'

The last one is the glasses application. You must have them connected via Wi-Fi, then as soon as you launch the application, select the glasses on the first screen and then on the second screen, go to Live and then Start. At the moment, I have commented out the part that gave YOLO labels and bounding boxes because I needed it for the video. If you need it, let me know and I'll see what needs to be uncommented (feel free to write to me).
