import odrive
import time



SERIAL_NUMBER_0 = "317532613431"
SERIAL_NUMBER_1 = "365A388C3131"

odrv0 = odrive.find_any(serial_number= SERIAL_NUMBER_0)
odrv1 = odrive.find_any(serial_number= SERIAL_NUMBER_1)

print(f"ODrive 0 found: {odrv0.serial_number}")
print(f"ODrive 1 found: {odrv1.serial_number}")

try:
    while True:
        # Read encoder positions from axis0 of both boards
        pos00 = odrv0.axis0.encoder.pos_estimate
        pos10 = odrv1.axis0.encoder.pos_estimate
        pos01 = odrv0.axis1.encoder.pos_estimate
        pos11 = odrv1.axis1.encoder.pos_estimate

        print(f"Axis0: {pos00:.3f} turns | : {pos10:.3f} turns")
        # print(f"Axis1: {pos01:.3f} turns | : {pos11:.3f} turns")
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Stopped by user.")
