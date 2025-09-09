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

    # Reset errors (no clear_errors() in firmware 0.5.4)
    axis.error = 0
    axis.motor.error = 0
    axis.encoder.error = 0

    axis.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE

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
        return True
    else:
        print(f"❌ {axis_name} calibration failed.")
        print(f"  Axis error: {axis.error}")
        print(f"  Motor error: {axis.motor.error}")
        print(f"  Encoder error: {axis.encoder.error}")
        return False

def main():
    odrv0 = connect_odrive()

    # Calibra solo axis0 (dato che axis1 non è collegato)
    success0 = run_calibration(odrv0.axis0, "Axis 0")
    # success1 = run_calibration(odrv0.axis1, "Axis 1")

    if success0:
        print("💾 Saving configuration...")
        try:
            odrv0.save_configuration()
        except Exception as e:
            print(f"⚠️ save_configuration ha causato un reboot (normale): {e}")
        print("✅ Done.")
    else:
        print("⚠️ Axis 0 failed calibration. Configuration NOT saved.")


if __name__ == '__main__':
    main()
