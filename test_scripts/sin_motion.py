import odrive
from odrive.enums import *
import time
import math

CONTROL_AXIS = 0  # Use axis0 or axis1

print("Connecting to ODrive...")
odrv = odrive.find_any(serial_number="317532613431")  # Change SN if needed
print("Connected!")

# 317532613431 left
# 365A388C3131 right
# 316E325A3431 new

# Select axis
# axis = odrv.axis0 if CONTROL_AXIS == 0 else odrv.axis1
axis = odrv.axis0

# Clear existing errors
print("Clearing errors...")
odrv.clear_errors()

# Start full calibration
print("Starting calibration...")
axis.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
while axis.current_state != AXIS_STATE_IDLE:
    time.sleep(0.1)

# Check for calibration success
if axis.error != 0 or axis.motor.error != 0 or axis.encoder.error != 0:
    print("❌ Error after calibration!")
    print("Axis error:    ", axis.error)
    print("Motor error:   ", axis.motor.error)
    print("Encoder error: ", axis.encoder.error)
    exit(1)

print("Wait for the beep!")
print("Calibration complete! Press Enter to continue...")
input()

# Check encoder readiness
if not axis.encoder.is_ready:
    print("❌ Encoder not ready after calibration!")
    exit(1)

# Configure controller for position control
axis.controller.config.control_mode = CONTROL_MODE_POSITION_CONTROL
axis.controller.config.input_mode = INPUT_MODE_PASSTHROUGH

# Enter closed-loop control
print("Entering closed-loop control...")
axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
time.sleep(0.2)

# Final check for errors
if axis.error != 0 or axis.motor.error != 0 or axis.encoder.error != 0:
    print("❌ Error entering closed-loop!")
    print("Axis error:    ", axis.error)
    print("Motor error:   ", axis.motor.error)
    print("Encoder error: ", axis.encoder.error)
    exit(1)

# Get starting position
start_pos = axis.encoder.pos_estimate
print(f"✅ Start position: {start_pos:.3f}")

print("\nPress Enter to begin movement...")
input()

# Sinusoidal position control loop
amp = 2.0      # Revolutions
freq = 0.2     # Hz
start_time = time.time()

try:
    while True:
        t = time.time() - start_time
        target = start_pos + amp * math.sin(2 * math.pi * freq * t)
        axis.controller.input_pos = target

        print(f"Target: {target:.2f}, Actual: {axis.encoder.pos_estimate:.2f}, Current: {axis.motor.current_control.Iq_measured:.2f}")
        time.sleep(0.01)

        # Monitor for runtime errors
        if axis.error != 0 or axis.motor.error != 0 or axis.encoder.error != 0:
            print("❌ Runtime errors detected:")
            print("Axis error:    ", axis.error)
            print("Motor error:   ", axis.motor.error)
            print("Encoder error: ", axis.encoder.error)
            break

except KeyboardInterrupt:
    print("\n🛑 Stopping...")
    axis.requested_state = AXIS_STATE_IDLE
    print("Axis set to IDLE.")
