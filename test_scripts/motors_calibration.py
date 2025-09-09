#!/usr/bin/env python3
import odrive
from odrive.enums import *
from time import sleep
import time


def connect_odrive():
    print("🔌 Searching for ODrive...")
    odrv0 = odrive.find_any()
    print(f"✅ Connected to ODrive serial: {odrv0.serial_number}")
    print(f"🔋 Bus voltage: {odrv0.vbus_voltage:.2f} V")
    return odrv0

def run_calibration(axis, axis_name):
    print(f"🧪 Starting calibration for {axis_name}...")

    axis.clear_errors()

    # Request full calibration
    axis.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE

    # Wait for completion
    timeout = 15  # seconds
    start_time = time.time()
    while axis.current_state != AXIS_STATE_IDLE:
        if time.time() - start_time > timeout:
            print(f"❌ {axis_name} calibration timeout!")
            break
        if axis.error != 0:
            print(f"❌ {axis_name} entered error state during calibration!")
            break
        sleep(0.1)

    if axis.error == 0:
        print(f"✅ {axis_name} calibration successful.")
        axis.motor.config.pre_calibrated = True
        axis.encoder.config.pre_calibrated = True
    else:
        print(f"❌ {axis_name} calibration failed.")
        print(f"  Axis error: {axis.error}")
        print(f"  Motor error: {axis.motor.error}")
        print(f"  Encoder error: {axis.encoder.error}")
        return False

    return True

def main():
    odrv0 = connect_odrive()
    
    # Run calibration on both axes
    success0 = run_calibration(odrv0.axis0, "Axis 0")
    # success1 = run_calibration(odrv0.axis1, "Axis 1")

    #if success0 and success1:
    if success0:
        print("💾 Saving configuration...")
        odrv0.save_configuration()
        print("🔄 Rebooting...")
        odrv0.reboot()
    else:
        print("⚠️ One or both axes failed calibration. Configuration NOT saved.")

if __name__ == '__main__':
    main()
