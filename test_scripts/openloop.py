import odrive
from odrive.enums import *
import time

# Connessione all'ODrive
print("Connecting to ODrive...")
odrv = odrive.find_any(serial_number="365A388C3131")  # Cambia SN se serve
print("Connected!")

# 317532613431 axis 1
# 365A388C3131 axis 0

# Seleziona l'asse da testare
axis = odrv.axis0  # o axis0

# Cancella errori
print("Clearing errors...")
odrv.clear_errors()

# Calibrazione motore (richiesta anche per test open-loop)
print("Calibrating motor...")
axis.requested_state = AXIS_STATE_MOTOR_CALIBRATION
while axis.current_state != AXIS_STATE_IDLE:
    time.sleep(0.1)

if axis.motor.error != 0:
    print(f"❌ Motor calibration failed! Error code: {axis.motor.error}")
    exit(1)

# Imposta controllo in torque (corrente)
axis.controller.config.control_mode = CONTROL_MODE_TORQUE_CONTROL
axis.controller.config.input_mode = INPUT_MODE_PASSTHROUGH

# Entra in closed-loop
axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
time.sleep(0.2)

print("Applying torque...")
# Applica coppia (corrente)
axis.controller.input_torque = 0.15
time.sleep(2)

axis.controller.input_torque = -0.15
time.sleep(2)

axis.controller.input_torque = 0.0
axis.requested_state = AXIS_STATE_IDLE
print("✅ Test completato.")
