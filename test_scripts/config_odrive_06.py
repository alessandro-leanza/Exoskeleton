#!/usr/bin/env python3
import odrive
from odrive.enums import *
from time import sleep

def connect_odrive():
    print("🔌 Searching for ODrive...")
    odrv0 = odrive.find_any()
    print(f"✅ Connected to ODrive serial: {odrv0.serial_number}")
    print(f"🔋 Bus voltage: {odrv0.vbus_voltage:.2f} V")
    return odrv0

def configure_odrive(odrv0):
    ''' 🚀 Motor Configuration '''
    for axis in [odrv0.axis0, odrv0.axis1]:
        axis.motor.config.current_lim = 45.0
        axis.motor.config.calibration_current = 5.0
        axis.motor.config.torque_constant = 0.0250606
        axis.motor.config.pole_pairs = 7
        axis.motor.config.torque_lim = 1.5
        axis.motor.config.pre_calibrated = False  # Set True later after calibration

    ''' 🎮 Controller Configuration '''
    for axis in [odrv0.axis0, odrv0.axis1]:
        axis.controller.config.control_mode = CONTROL_MODE_TORQUE_CONTROL  # = 1
        axis.controller.config.input_mode = INPUT_MODE_PASSTHROUGH          # = 1
        axis.controller.config.vel_limit = 10
        axis.controller.config.pos_gain = 20.0
        axis.controller.config.vel_gain = 0.16666667
        axis.controller.config.vel_integrator_gain = 0.33333334

    ''' 🎯 Encoder Configuration '''
    odrv0.config.gpio5_mode = GPIO_MODE_DIGITAL  # 0
    odrv0.config.gpio6_mode = GPIO_MODE_DIGITAL
    odrv0.axis0.encoder.config.abs_spi_cs_gpio_pin = 5
    odrv0.axis1.encoder.config.abs_spi_cs_gpio_pin = 6

    for axis in [odrv0.axis0, odrv0.axis1]:
        axis.encoder.config.cpr = 16384
        axis.encoder.config.mode = ENCODER_MODE_SPI_ABS_AMS
        axis.encoder.config.use_index = False
        axis.encoder.config.pre_calibrated = False  # Set True later after calibration

    ''' 🛑 Disable auto-calibration (you will run it manually) '''
    for axis in [odrv0.axis0, odrv0.axis1]:
        axis.config.startup_motor_calibration = False
        axis.config.startup_encoder_offset_calibration = False
        axis.config.startup_closed_loop_control = False
        axis.config.startup_encoder_index_search = False
        axis.config.startup_homing = False

    sleep(0.1)

    print("✅ Configuration applied.")
    ans = input("💾 Save configuration and reboot? [y/N]: ")

    if ans.upper() == "Y":
        try:
            odrv0.save_configuration()
        except:
            print("ODrive rebooted after saving. Waiting to reconnect...")
            sleep(5)
            odrv0 = odrive.find_any()
            print("Reconnected to {}".format(hex(odrv0.serial_number)))

if __name__ == '__main__':
    odrv = connect_odrive()
    configure_odrive(odrv)
