from setuptools import setup

package_name = 'exo_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Exoskeleton control package with C++ and Python nodes.',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'twoboards_odrive = exo_control.twoboards_odrive:main',
            'admittancecontrol_box = exo_control.admittancecontrol_box:main',
            'stoop_trajs = exo_control.stoop_trajs:main',
            'yolo_grasp_detector = exo_control.yolo_grasp_detector:main',
            'nominal_traj = exo_control.nominal_traj:main',
            'traj_client = exo_control.traj_client:main',
            'glasses_yolo_detector = exo_control.glasses_yolo_detector:main',
            'rel_no_vision_trigger = exo_control.rel_no_vision_trigger:main',
            'ass_no_vision_trigger = exo_control.ass_no_vision_trigger:main',
            'realsense_yolo = exo_control.realsense_yolo:main',
        ],
    },
)
