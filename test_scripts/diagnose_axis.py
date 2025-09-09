import odrive
from odrive.enums import *
import time


SERIAL_0 = "317532613431"
SERIAL_1 = "365A388C3131"

def get_param_safe(obj, param_name):
    try:
        return obj._channel.get_float(f"{obj._path}.{param_name}")
    except Exception as e:
        return f"<ERR: {e}>"

def diagnose_axis(odrv, axis_name, axis):
    print(f"\n🔍 Diagnosing {axis_name}")

    try:
        print(f"  ↪ Axis error:    {axis.error}")
        print(f"  ↪ Motor error:   {axis.motor.error}")
        print(f"  ↪ Encoder error: {axis.encoder.error}")
        print(f"  ↪ Is calibrated: {axis.motor.is_calibrated}")
        print(f"  ↪ Encoder ready: {axis.encoder.is_ready}")

        # Access motor parameters via get_param_safe (firmware 0.5.x workaround)
        R = get_param_safe(axis.motor, "config.phase_resistance")
        L = get_param_safe(axis.motor, "config.phase_inductance")
        Ib = get_param_safe(axis.motor, "current_meas_phB")
        Ic = get_param_safe(axis.motor, "current_meas_phC")

        print(f"  ↪ Phase resistance: {R}")
        print(f"  ↪ Phase inductance: {L}")
        print(f"  ↪ Current phB: {Ib} A")
        print(f"  ↪ Current phC: {Ic} A")

        if isinstance(R, float) and R < 0.01:
            print("  ⚠️ WARNING: Phase resistance too low — motor likely not connected!")

    except Exception as e:
        print(f"  ❌ ERROR: Failed to access {axis_name}: {e}")


def main():
    print("Connecting to both ODrives...")
    odrv0 = odrive.find_any(serial_number=SERIAL_0)
    odrv1 = odrive.find_any(serial_number=SERIAL_1)
    print("✅ Connected to both ODrives.")

    print("Clearing errors...")
    odrv0.clear_errors()
    odrv1.clear_errors()

    diagnose_axis(odrv0, "odrv0.axis0", odrv0.axis0)
    diagnose_axis(odrv0, "odrv0.axis1", odrv0.axis1)
    diagnose_axis(odrv1, "odrv1.axis0", odrv1.axis0)
    diagnose_axis(odrv1, "odrv1.axis1", odrv1.axis1)


if __name__ == "__main__":
    main()
