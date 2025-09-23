#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import math
import numpy as np
import rclpy.serialization as rclpy_ser
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.animation import FuncAnimation

# Messaggi
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float32MultiArray

# ====== Parametri di default ======
DEFAULT_BAG_DIR = os.environ.get(
    "ROSBAG_DIR",
    str((Path(__file__).parent / "rosbags" / "z_vis_4").resolve())
)
DEFAULT_OUT = "plot.png"

# ====== Finestre temporali (modifica qui o via CLI) ======
T_IN_S  = 0.0
T_END_S = 80.0

# ====== Soglie orizzontali su theta (in radianti) ======
THETA_STAND_RAD = 0.05
THETA_BEND_RAD  = 0.70

# ====== Colori stile “vecchio script” ======
POS_COLOR       = "blue"     # posizione
TAU_W_COLOR     = "red"      # tau_w
TAU_BOX_COLOR   = "gold"     # tau_box
TAU_ASS_COLOR   = "orange"   # tau_ass (total)
MEAS_COLOR      = "tab:gray" # tau_meas (libero)

# ====== Utility Rosbag2 ======
def open_bag(bag_uri: str) -> SequentialReader:
    p = Path(bag_uri).expanduser().resolve()
    meta = p / "metadata.yaml"
    if not meta.exists():
        raise FileNotFoundError(
            f"Cartella bag non trovata o invalida: {p}\nAtteso: {meta}"
        )
    reader = SequentialReader()
    storage_options = StorageOptions(uri=str(p), storage_id="sqlite3")
    converter_options = ConverterOptions(input_serialization_format="cdr",
                                         output_serialization_format="cdr")
    reader.open(storage_options, converter_options)
    return reader

def iterate_bag(reader: SequentialReader):
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        yield topic, int(t), data

# ====== Raccolta segnali ======
def collect_signals(bag_dir: str):
    reader = open_bag(bag_dir)
    t0_ns = None
    series = {
        "theta_w": [],        # JointState.position[0]
        "tau_w": [],          # Float64
        "tau_box": [],        # Float64
        "tau_ass_total": [],  # Float64
        "tau_meas": [],       # Float32MultiArray.data[0]
    }
    def append(name: str, t_ns: int, val: float):
        nonlocal t0_ns
        if t0_ns is None:
            t0_ns = t_ns
        t_rel = (t_ns - t0_ns) * 1e-9
        series[name].append((t_rel, float(val)))
    for topic, stamp_ns, raw in iterate_bag(reader):
        try:
            if topic == "/joint_states":
                msg = rclpy_ser.deserialize_message(raw, JointState)
                if msg.position and len(msg.position) > 0:
                    append("theta_w", stamp_ns, msg.position[0])
            elif topic == "/admittance/tau_w":
                msg = rclpy_ser.deserialize_message(raw, Float64)
                append("tau_w", stamp_ns, msg.data)
            elif topic == "/admittance/tau_box":
                msg = rclpy_ser.deserialize_message(raw, Float64)
                append("tau_box", stamp_ns, msg.data)
            elif topic == "/admittance/tau_ass_total":
                msg = rclpy_ser.deserialize_message(raw, Float64)
                append("tau_ass_total", stamp_ns, msg.data)
            elif topic == "/admittance/tau_meas":
                msg = rclpy_ser.deserialize_message(raw, Float32MultiArray)
                if msg.data and len(msg.data) > 0:
                    append("tau_meas", stamp_ns, msg.data[0])
        except Exception:
            pass
    return series

def crop_and_shift(series: dict, t_in: float, t_end: float) -> dict:
    out = {}
    for name, arr in series.items():
        cropped = []
        for (t, y) in arr:
            if t_in <= t <= t_end:
                cropped.append((t - t_in, y))
        out[name] = cropped
    return out

# ====== Plot + (opzionale) animazione ======
def plot_series(series: dict, out_path: str, animate: bool, fps: int):
    def unzip(name):
        if not series[name]:
            return np.array([]), np.array([])
        t, y = zip(*series[name])
        return np.asarray(t, dtype=float), np.asarray(y, dtype=float)

    t_theta, theta = unzip("theta_w")
    t_tw, tw = unzip("tau_w")
    t_tb, tb = unzip("tau_box")
    t_tass, tass = unzip("tau_ass_total")
    t_tm, tm = unzip("tau_meas")

    # Figura: 4 subplot verticali
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    ax1, ax2, ax3, ax4 = axes

    # 1) Lumbar Position in gradi
    if t_theta.size:
        ax1.plot(t_theta, np.degrees(theta), color=POS_COLOR, label=r"$\theta_w$")
    th_stand_deg = math.degrees(THETA_STAND_RAD)
    th_bend_deg  = math.degrees(THETA_BEND_RAD)
    ax1.axhline(th_bend_deg, color="k", linestyle="-", linewidth=1.5,
                label=rf"$\theta_{{\mathrm{{bend}}}}={th_bend_deg:.0f}^\circ$")
    ax1.axhline(th_stand_deg, color="k", linestyle="--", linewidth=1.5,
                label=rf"$\theta_{{\mathrm{{stand}}}}={th_stand_deg:.0f}^\circ$")
    ax1.set_title("Lumbar Position")
    ax1.set_ylabel(r"$\theta$ [deg]")
    ax1.grid(True)
    ax1.legend(loc="best")
    ax1.yaxis.set_major_locator(MultipleLocator(10))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    # 2) Torque Contributions
    if t_tw.size:
        ax2.plot(t_tw, tw, color=TAU_W_COLOR, label=r"$\tau_w$")
    if t_tb.size:
        ax2.plot(t_tb, tb, color=TAU_BOX_COLOR, label=r"$\tau_{\mathrm{box}}$")
    ax2.set_title("Torque Contributions")
    ax2.set_ylabel("Torque [Nm]")
    ax2.grid(True)
    ax2.legend(loc="best")

    # 3) Assistance Torque (total)
    if t_tass.size:
        ax3.plot(t_tass, tass, color=TAU_ASS_COLOR, label=r"$\tau_{\mathrm{ass}}$")
    ax3.set_title("Assistance Torque")
    ax3.set_ylabel("Torque [Nm]")
    ax3.grid(True)
    ax3.legend(loc="best")

    # 4) Measured Torque
    if t_tm.size:
        ax4.plot(t_tm, tm, color=MEAS_COLOR, label=r"$\tau_{\mathrm{meas}}$")
    ax4.set_title("Measured Torque")
    ax4.set_ylabel("Torque [Nm]")
    ax4.set_xlabel("t [s]   (t=0 → Tin)")
    ax4.grid(True)
    ax4.legend(loc="best")

    fig.tight_layout()

    # Salva PNG statico
    out_path = str(Path(out_path).expanduser().resolve())
    fig.savefig(out_path, dpi=150)
    print(f"[OK] Plot salvato in: {out_path}")

    # ===== Animazione “pallina” in tempo reale =====
    if animate:
        # timeline comune (durata = max t disponibile)
        t_max = 0.0
        for t in (t_theta, t_tw, t_tb, t_tass, t_tm):
            if t.size:
                t_max = max(t_max, t.max())
        if t_max <= 0.0:
            print("[ANIM] Nessun dato utile per animare.")
            plt.show()
            return

        # Campionamento a fps costante
        dt = 1.0 / max(1, fps)
        t_anim = np.arange(0.0, t_max, dt)

        # Interpola theta_w (in gradi) e tau_ass_total sui tempi uniformi
        def interp(ts, ys, tgrid, default=np.nan):
            if ts.size == 0:
                return np.full_like(tgrid, default, dtype=float)
            # clamp bordi per evitare NaN
            return np.interp(tgrid, ts, ys, left=ys[0], right=ys[-1])

        theta_deg = interp(t_theta, np.degrees(theta) if t_theta.size else np.array([]), t_anim, default=np.nan)
        tass_val  = interp(t_tass, tass if t_tass.size else np.array([]), t_anim, default=np.nan)

        # Oggetti grafici “pallina” + cursore verticale
        ball1, = ax1.plot([], [], marker='o', markersize=8, linestyle='None', color='tab:red')
        ball3, = ax3.plot([], [], marker='o', markersize=8, linestyle='None', color='tab:red')
        vline_color = (0, 0, 0, 0.25)
        vline1 = ax1.axvline(0.0, color=vline_color)
        vline2 = ax2.axvline(0.0, color=vline_color)
        vline3 = ax3.axvline(0.0, color=vline_color)
        vline4 = ax4.axvline(0.0, color=vline_color)

        # Limiti Y per evitare rescale durante animazione
        for ax in (ax1, ax2, ax3, ax4):
            ax.relim(); ax.autoscale_view()

        def init():
            ball1.set_data([], [])
            ball3.set_data([], [])
            return ball1, ball3, vline1, vline2, vline3, vline4

        def update(i):
            t = t_anim[i]
            # aggiorna pallina 1 (theta_w)
            if not np.isnan(theta_deg[i]):
                ball1.set_data([t], [theta_deg[i]])
            # aggiorna pallina 3 (tau_ass)
            if not np.isnan(tass_val[i]):
                ball3.set_data([t], [tass_val[i]])
            # cursori verticali su tutti i subplot
            for vl in (vline1, vline2, vline3, vline4):
                vl.set_xdata([t, t])
            return ball1, ball3, vline1, vline2, vline3, vline4

        interval_ms = int(1000.0 / max(1, fps))  # così 1 ciclo ~ 1/fps secondi → durata = t_max “reale”
        anim = FuncAnimation(fig, update, frames=len(t_anim), init_func=init,
                             interval=interval_ms, blit=True, repeat=False)
        # Nota: non salvo il video di default. Se vuoi l’MP4 chiedimelo e aggiungo writer/ffmpeg.

    # Mostra finestra interattiva (sempre)
    plt.show()

# ====== CLI ======
def parse_args():
    ap = argparse.ArgumentParser(
        description="Estrae segnali da rosbag2, taglia su [Tin,Tend], shifta t e plotta 4 subplot (+ animazione opzionale).")

    ap.add_argument("bag_dir", nargs="?", default=DEFAULT_BAG_DIR,
                    help="Cartella del rosbag2 (default: ENV ROSBAG_DIR o rosbags/z_novis accanto allo script).")
    ap.add_argument("-o", "--out", default=DEFAULT_OUT,
                    help=f"Percorso immagine di output (default: {DEFAULT_OUT}).")
    ap.add_argument("--tin", type=float, default=T_IN_S, help=f"Inizio finestra [s] (default: {T_IN_S})")
    ap.add_argument("--tend", type=float, default=T_END_S, help=f"Fine finestra [s] (default: {T_END_S})")

    # ✅ animate di default ON, con switch per spegnerla
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--animate", dest="animate", action="store_true", help="(default)")
    group.add_argument("--no-animate", dest="animate", action="store_false", help="Disabilita animazione")
    ap.set_defaults(animate=True)

    ap.add_argument("--fps", type=int, default=30, help="FPS animazione (default: 30).")
    return ap.parse_args()


def main():
    args = parse_args()
    bag_dir = Path(args.bag_dir).expanduser().resolve()
    print(f"[INFO] Uso bag_dir = {bag_dir}")
    print(f"[INFO] Finestra = [{args.tin:.3f}, {args.tend:.3f}] s  (t=0 → Tin)")
    raw = collect_signals(str(bag_dir))
    sliced = crop_and_shift(raw, args.tin, args.tend)
    plot_series(sliced, args.out, animate=args.animate, fps=args.fps)

if __name__ == "__main__":
    main()
