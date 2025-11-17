#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---- File attesi (da: ros2 topic echo <topic> --csv > file.csv) ----
F_THETA_REF = 'theta_ref.csv'        # Float64 -> una colonna (valori)
F_JOINT     = 'joint_states.csv'     # JointState -> 11 colonne: sec,nsec,frame_id, joint_0, joint_1, pos0,pos1,vel0,vel1,eff0,eff1
F_TAU_MEAS  = 'torque_estimated.csv' # Float32MultiArray -> 2+ colonne (data[0], data[1], ...)
F_POS_CMD   = 'position_cmd.csv'     # Float32MultiArray -> 2+ colonne (data[0], data[1], ...)

JOINT_IDX = 1  # 0 o 1

def must_exist(p: str):
    if not Path(p).exists():
        raise FileNotFoundError(f"File non trovato: {p}")
    return p

# ---- joint_states: lettura robusta ----
must_exist(F_JOINT)
raw = pd.read_csv(F_JOINT, header=None)

# ci aspettiamo almeno 10 colonne; nel tuo caso 11 (con frame_id)
if raw.shape[1] < 10:
    raise ValueError(f"{F_JOINT} ha solo {raw.shape[1]} colonne; attese >=10.")

# prime due = tempo
sec  = raw.iloc[:, 0].astype(float).to_numpy()
nsec = raw.iloc[:, 1].astype(float).to_numpy()
t_js = (sec + nsec * 1e-9) - (sec[0] + nsec[0] * 1e-9)

# rimuovi colonne non numeriche: frame_id (col 2), joint_0 (col 3), joint_1 (col 4)
# poi prendi le ultime 6 (pos0,pos1,vel0,vel1,eff0,eff1)
# Nota: se ci fossero più colonne, cerchiamo direttamente le ultime 6 numeriche
numeric_cols = raw.select_dtypes(include=[np.number])
if numeric_cols.shape[1] < 8:
    # fallback: drop col 2-4 “a mano” e riprendi
    data_part = raw.drop(columns=[2,3,4], errors='ignore')
    numeric_cols = data_part.select_dtypes(include=[np.number])

# le prime due numeriche sono tempo; le successive 6 sono dati
if numeric_cols.shape[1] < 8:
    raise ValueError("Impossibile isolare le 6 colonne numeriche di pos/vel/eff.")
num = numeric_cols.to_numpy()
# Trova le 6 colonne finali dopo tempo: (assumiamo ordine stabile)
pos0, pos1, vel0, vel1, eff0, eff1 = num[:, -6:].T

# scegli giunto
theta     = pos1 if JOINT_IDX == 1 else pos0
theta_dot = vel1 if JOINT_IDX == 1 else vel0
tau_jnt   = eff1 if JOINT_IDX == 1 else eff0

# ---- altri topic (senza header, senza timestamp) ----
must_exist(F_THETA_REF)
theta_ref_vals = pd.read_csv(F_THETA_REF, header=None).iloc[:,0].astype(float).to_numpy()

must_exist(F_TAU_MEAS)
tau_meas_df = pd.read_csv(F_TAU_MEAS, header=None)
if tau_meas_df.shape[1] <= JOINT_IDX:
    raise ValueError(f"{F_TAU_MEAS} ha {tau_meas_df.shape[1]} colonne, JOINT_IDX={JOINT_IDX}.")
tau_meas_vals = tau_meas_df.iloc[:, JOINT_IDX].astype(float).to_numpy()

must_exist(F_POS_CMD)
pos_cmd_df = pd.read_csv(F_POS_CMD, header=None)
if pos_cmd_df.shape[1] <= JOINT_IDX:
    raise ValueError(f"{F_POS_CMD} ha {pos_cmd_df.shape[1]} colonne, JOINT_IDX={JOINT_IDX}.")
theta_cmd_vals = pos_cmd_df.iloc[:, JOINT_IDX].astype(float).to_numpy()

# ---- costruisci assi temporali sintetici per gli altri topic e interpola su t_js ----
def stretch_time_like(length, t0, t1):
    if length <= 1 or t1 <= t0: return np.arange(length, dtype=float)
    return np.linspace(t0, t1, num=length)

t0, t1 = float(t_js[0]), float(t_js[-1])
t_theta = stretch_time_like(len(theta_ref_vals), t0, t1)
t_tau   = stretch_time_like(len(tau_meas_vals),  t0, t1)
t_cmd   = stretch_time_like(len(theta_cmd_vals), t0, t1)

theta_ref_interp = np.interp(t_js, t_theta, theta_ref_vals)
tau_meas_interp  = np.interp(t_js, t_tau,   tau_meas_vals)
theta_cmd_interp = np.interp(t_js, t_cmd,   theta_cmd_vals)

# ---- grandezze derivate/plot ----
error = theta - theta_ref_interp
power = tau_meas_interp * theta_dot  # W

plt.figure(figsize=(8,4))
plt.plot(t_js, theta_ref_interp, label=r'$\theta_{ref}$')
plt.plot(t_js, theta, label=r'$\theta$ (joint)')
plt.xlabel('Time [s]'); plt.ylabel('Angle [rad]')
plt.title('Reference tracking'); plt.grid(True); plt.legend()

plt.figure(figsize=(8,3))
plt.plot(t_js, error)
plt.xlabel('Time [s]'); plt.ylabel('Error [rad]')
plt.title('Tracking error'); plt.grid(True)

plt.figure(figsize=(8,4))
plt.plot(t_js, tau_meas_interp, label=r'$\tau_{meas}$')
plt.xlabel('Time [s]'); plt.ylabel('Torque [Nm]')
plt.title('Measured torque'); plt.grid(True); plt.legend()

plt.figure(figsize=(8,3))
plt.plot(t_js, power)
plt.xlabel('Time [s]'); plt.ylabel('Power [W]')
plt.title('Estimated power exchange ($\\tau_{meas}\\cdot\\dot{\\theta}$)'); plt.grid(True)

plt.tight_layout()
plt.show()
