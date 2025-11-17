#!/usr/bin/env python3
import os
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# === Config ===
BAG_DIR = "try_3"  # cartella della bag (quella che hai creato con -o try_3)
TOPICS = [
    "/joint_states",
    "/admittance/tau_box",
    "/admittance/tau_ass_total",
    "/admittance/tau_meas",
    "/admittance/tau_w",
]

# === Lettore rosbag2 ===
storage_options = StorageOptions(uri=BAG_DIR, storage_id="sqlite3")
converter_options = ConverterOptions(input_serialization_format="cdr",
                                     output_serialization_format="cdr")
reader = SequentialReader()
reader.open(storage_options, converter_options)

# Mappa topic -> (type_name, msg_cls)
topic_info = {}
for t in reader.get_all_topics_and_types():
    topic_info[t.name] = {
        "type": t.type,
        "cls": get_message(t.type)
    }

# Strutture per accumulo dati
data = defaultdict(list)  # per DataFrame finale

def append_sample(ts_ns, name, value):
    # ts in secondi float
    data["timestamp_s"].append(ts_ns / 1e9)
    data["signal"].append(name)
    data["value"].append(float(value))

# Loop lettura messaggi
while reader.has_next():
    topic, raw, ts = reader.read_next()
    if topic not in TOPICS:
        continue

    msg_cls = topic_info[topic]["cls"]
    msg = deserialize_message(raw, msg_cls)

    # Estrai i campi richiesti
    if topic == "/joint_states":
        # sensor_msgs/JointState: prendiamo position[0] se disponibile
        val = None
        try:
            if msg.position and len(msg.position) > 0:
                val = msg.position[0]
        except Exception:
            val = None
        if val is not None:
            append_sample(ts, "joint_0_position", val)

    elif topic in ("/admittance/tau_box", "/admittance/tau_ass_total", "/admittance/tau_w"):
        # std_msgs/Float64 -> .data
        try:
            append_sample(ts, topic.split("/")[-1], msg.data)
        except Exception:
            pass

    elif topic == "/admittance/tau_meas":
        # std_msgs/Float32MultiArray -> .data[0]
        try:
            if msg.data and len(msg.data) > 0:
                append_sample(ts, "tau_meas_0", msg.data[0])
        except Exception:
            pass

# Se non è arrivato nulla, esci
if not data:
    raise SystemExit("Nessun dato letto. Controlla percorso bag e nomi topic.")

# Costruisci DataFrame e “pivot” per avere colonne per ogni segnale
df_long = pd.DataFrame(data)
# Sincronizza le serie per time merging: creiamo un indice tempo unificato
df_wide = df_long.pivot_table(index="timestamp_s", columns="signal", values="value", aggfunc="last")
df_wide.sort_index(inplace=True)

# Opzionale: fill dei buchi per grafico più “pulito”
df_wide_interp = df_wide.interpolate(method="time", limit_direction="both")

# Salva CSV
csv_path = os.path.join(BAG_DIR, "signals.csv")
df_wide_interp.to_csv(csv_path, float_format="%.6f")
print(f"CSV salvato in: {csv_path}")

# Plot
plt.figure(figsize=(12, 6))
# Colonne attese in ordine richiesto; plottiamo solo quelle presenti
ordered_cols = [
    "joint_0_position",     # /joint_states/position[0]
    "tau_box",              # /admittance/tau_box/data
    "tau_ass_total",        # /admittance/tau_ass_total/data
    "tau_meas_0",           # /admittance/tau_meas/data[0]
    "tau_w",                # /admittance/tau_w/data
]
present_cols = [c for c in ordered_cols if c in df_wide_interp.columns]

for c in present_cols:
    plt.plot(df_wide_interp.index, df_wide_interp[c], label=c)

plt.xlabel("time [s]")
plt.ylabel("value")
plt.title("Exoskeleton signals")
plt.legend()
plt.grid(True)
png_path = os.path.join(BAG_DIR, "plot_exo_signals.png")
plt.tight_layout()
plt.savefig(png_path, dpi=150)
print(f"Plot salvato in: {png_path}")
