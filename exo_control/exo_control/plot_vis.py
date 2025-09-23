#!/usr/bin/env python3
# === CONFIGURAZIONE ===
BAG_DIR   = "rosbags/rosbag2_2025_09_19-11_12_51"   # cartella che contiene signals.csv rosbag2_2025_09_18-12_13_02 = z_vision
START_S   = 122    # inizio finestra [s] relativo all'inizio del file (None = inizio)
END_S     = 133     # fine finestra [s]   relativo all'inizio del file (None = fine)

THETA_MAX_TARGET = 1.1   # [rad] nuovo massimo desiderato nella finestra
SHIFT_THETA_DEG = -3.5  # Offset verticale della posizione (in GRADI): negativo = verso il basso
ALLOW_EXPANSION  = True   # True: consenti anche aumenti; False: solo compressione
SHOW_ORIG_THETA  = False  # True per sovrapporre anche la traccia originale (opzionale)

# Soglie orizzontali (in GRADI) + switch plotting in gradi
THETA_BEND_DEG   = 52.0   # livello "bend" (gradi)
THETA_STAND_DEG  = 5.0    # livello "stand" (gradi)
SHOW_THETA_BEND  = True
SHOW_THETA_STAND = True
SHOW_POS_IN_DEG  = True   # se True, asse posizione in gradi (calcoli restano in radianti)

MARKER_LINESTYLE = ":"
MARKER_COLOR     = "#067c00"   # azzurrino discreto
MARKER_ALPHA     = 0.8
LABEL_FONTSIZE   = 10

# Tempi schedule (inizializzazioni/definiti da incroci più sotto)
T0  = 0.0
T01 = None   # quando theta_w == theta_stand (primo crossing)
T10 = None   # quando theta_w == theta_bend  (primo crossing)
T1  = 3.5   # fisso
T1w = None   # quando theta_w == theta_bend dopo T1
T2  = None   # quando theta_w == theta_stand dopo T1w
T3  = None   # quando theta_w == theta_stand dopo T2
T4  = None   # quando theta_w == theta_bend  dopo T3
T5  = 9.0  # fisso
T5w = None   # quando theta_w == theta_bend dopo T5
T6  = None   # quando theta_w == theta_stand dopo T5w

SHOW_FIG  = True

# === PARAMETRI MODELLO (come nel controller) ===
h = 1.80
w = 70.0
g       = 9.81
m_w     = (1-0.142-0.1*2-0.06*2) * w #    0.538 * w  
l_w     = (0.72-0.53) * h # 0.
m_b     = 4.0
l_int   = 0.55
l_b     = 0.20
offset  = 0.0

coeff_assist   = 1.0
assist_max_nm  = 200.0  # LIMITE MECCANICO: SOLO visualizzazione "applied" (simulazione motore)

# Rampa tau_ass (come nel nodo)
RAMP_S   = 0.3   # [s] durata rampa (tau_time_set)
RAMP_EPS = 0.1   # [Nm] soglia per rischedulare la rampa (tau_resched_eps)

# === SCRIPT ===
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib as mpl
from typing import List, Optional
# ROS 2 bag reader
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
from matplotlib.animation import FuncAnimation, FFMpegWriter




mpl.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
})

mpl.rcParams.update({
    # dimensioni
    "axes.titlesize": 9,        # titoli dei subplot
    "figure.titlesize": 10,      # suptitle (se usi fig.suptitle)

    # famiglia/peso/stile font di default
    "font.family": "DejaVu Sans",   # o "Liberation Sans", "Arial", ecc.
    "font.size": 11,
})
mpl.rcParams["mathtext.fontset"] = "stix"

TORQUE_TICK_NM = 10  # passo delle tacche Y per le coppie
TIME_TICK_S = 2

# Palette colori
POS_COLOR        = "blue"      # posizione
TAU_W_COLOR      = "red"       # tau_w
TAU_BOX_COLOR    = "gold"      # tau_box
TAU_ASS_COLOR    = "orange"    # tau_ass with vision (APPLIED)
TAU_NOVIS_COLOR  = "crimson"   # tau_ass no_vision (APPLIED)

CSV_NAME = "signals.csv"
OUT_NAME = "custom_plot.png"

OUT_MP4_NAME = "custom_plot.mp4"  # nome file video da salvare
MP4_DPI      = 150                # risoluzione del render
MP4_BITRATE  = 4000               # kbps, bilancia qualità/peso


def load_signals(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV non trovato: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)
    if df.empty:
        raise ValueError("CSV vuoto.")
    # Ordina ed elimina duplicati indice per robustezza
    df = df[~df.index.duplicated(keep='first')].sort_index()
    return df

def index_to_seconds(df: pd.DataFrame) -> np.ndarray:
    idx = df.index
    # tenta datetime
    try:
        dt = pd.to_datetime(idx)
        if not np.any(pd.isna(dt)):
            return (dt - dt[0]).total_seconds().astype(float)
    except Exception:
        pass
    # altrimenti numerico
    try:
        t = idx.astype(float)
        return t - float(t[0])
    except Exception as e:
        raise ValueError("Indice tempo non interpretabile come datetime o secondi float.") from e


def load_joint0_from_bag(bag_dir: str,
                         topic_name: str = "/joint_states",
                         joint_index: int = 0) -> pd.DataFrame:
    """
    Legge /joint_states dalla rosbag e restituisce un DataFrame con:
      index: t [s] (shiftato da t0=primo timestamp)
      column: 'joint_0_position' (position[joint_index])
    """
    if not os.path.isdir(bag_dir):
        raise FileNotFoundError(f"Cartella bag non trovata: {bag_dir}")

    # Proviamo prima sqlite3 poi mcap (in automatico)
    def _open_reader(storage_id: str):
        reader = SequentialReader()
        storage = StorageOptions(uri=bag_dir, storage_id=storage_id)
        converter = ConverterOptions("", "")
        reader.open(storage, converter)
        return reader

    reader = None
    try:
        reader = _open_reader("sqlite3")
    except Exception:
        # fallback MCAP
        reader = _open_reader("mcap")

    topics_and_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topics_and_types}

    if topic_name not in type_map:
        raise RuntimeError(f"Topic '{topic_name}' non trovato nella bag. Presenti: {list(type_map.keys())}")

    msg_type = get_message(type_map[topic_name])

    times_ns = []
    values = []

    t0_ns = None
    while reader.has_next():
        (topic, data, stamp) = reader.read_next()
        if topic != topic_name:
            continue
        msg = deserialize_message(data, msg_type)
        try:
            pos = float(msg.position[joint_index])
        except Exception:
            continue

        if t0_ns is None:
            t0_ns = stamp
        t_rel_s = (stamp - t0_ns) * 1e-9  # ns -> s relativo
        times_ns.append(t_rel_s)
        values.append(pos)

    if len(times_ns) < 2:
        raise RuntimeError("Pochi campioni letti dal topic /joint_states.")

    df = pd.DataFrame({"t_s": np.array(times_ns, dtype=float),
                       "joint_0_position": np.array(values, dtype=float)})
    df = df.set_index("t_s")
    return df

def slice_time_window(t, df, t_start, t_end):
    if t_start is None: t_start = float(t[0])
    if t_end   is None: t_end   = float(t[-1])
    if t_end < t_start:
        raise ValueError(f"t_end ({t_end}) < t_start ({t_start})")
    mask = (t >= t_start) & (t <= t_end)
    if mask.sum() < 2:
        raise ValueError("Finestra temporale troppo stretta o fuori range: pochi campioni.")
    return (t[mask] - float(t[mask][0])), df.loc[mask]

def compute_tau_w(theta):
    return m_w * g * l_w * np.sin(theta + offset)

def compute_tau_box(theta):
    return m_b * g * (l_int * np.sin(theta + offset) + l_b)

def clamp(x, low, high):
    return np.minimum(np.maximum(x, low), high)

def schedule_weights(t_rel: np.ndarray, full_box_seg34: bool = True):
    """
    Pesi (w_w, w_box):
      [T0,T1): 0               -> (0,0)
      [T1,T2): tau_w+tau_box   -> (1,1)
      [T2,T3): 0               -> (0,0)
      [T3,T4): tau_box         -> (0,1) se full_box_seg34=True, altrimenti (0,0.5)
      [T4,T5): 0               -> (0,0)
      [T5,T6]: tau_w           -> (1,0)
      Fuori da [T0,T6] -> (0,0)
    (Robusta a T* = None)
    """
    w_w = np.zeros_like(t_rel, dtype=float)
    w_b = np.zeros_like(t_rel, dtype=float)

    def seg_mask(a, b):
        if a is None or b is None:
            return np.zeros_like(t_rel, dtype=bool)
        return (t_rel >= a) & (t_rel < b)

    seg12 = seg_mask(T1, T2)
    w_w[seg12] = 0.25; w_b[seg12] = 0.25

    seg34 = seg_mask(T3, T4)
    w_w[seg34] = 0.0
    w_b[seg34] = 0.25 if full_box_seg34 else 0.25

    if (T5 is not None) and (T6 is not None):
        seg56 = (t_rel >= T5) & (t_rel <= T6)
    else:
        seg56 = np.zeros_like(t_rel, dtype=bool)
    w_w[seg56] = 0.25; w_b[seg56] = 0.0

    return w_w, w_b


def apply_event_ramps(recipe: np.ndarray,
                      t_rel: np.ndarray,
                      rise_times: List[Optional[float]],
                      fall_times: List[Optional[float]],
                      ramp_s: float,
                      resched_eps: float = RAMP_EPS) -> np.ndarray:
    """
    Rampa event-driven SENZA LIMITE:
      - ai rise_times: rampa verso recipe(t_event) (raw, no clamp)
      - ai fall_times: rampa verso 0
      - durante la rampa: se |recipe - target| > resched_eps → ripianifica verso il nuovo target
      - dopo la rampa: tracking della ricetta live
    Ignora eventi None.
    """
    y = np.zeros_like(recipe, dtype=float)
    if recipe.size == 0:
        return y

    rises = sorted([e for e in (rise_times or []) if e is not None])
    falls = sorted([e for e in (fall_times or []) if e is not None])

    def crossed(tp, t, evlist):
        for e in evlist:
            if tp < e <= t:
                return e
        return None

    ramp_active = False
    y_start = 0.0
    y_end   = 0.0
    t0 = t1 = t_rel[0]
    last_event = None

    y[0] = 0.0

    for i in range(1, len(t_rel)):
        tp, t = t_rel[i-1], t_rel[i]

        ev_r = crossed(tp, t, rises)
        ev_f = crossed(tp, t, falls)

        # priorità: rise > fall se coincidono
        if ev_r is not None:
            y_curr = y[i-1]
            target = recipe[i]   # raw target (no clamp)
            t0 = ev_r
            t1 = ev_r + max(0.05, ramp_s)
            y_start, y_end = y_curr, target
            ramp_active = True
            last_event = "rise"

        elif ev_f is not None:
            y_curr = y[i-1]
            target = 0.0
            t0 = ev_f
            t1 = ev_f + max(0.05, ramp_s)
            y_start, y_end = y_curr, target
            ramp_active = True
            last_event = "fall"

        if ramp_active:
            if t >= t1:
                y[i] = y_end
                ramp_active = False
            else:
                u = max(0.0, min(1.0, (t - t0) / (t1 - t0)))
                s = u*u*(3.0 - 2.0*u)
                y_curr = y_start + (y_end - y_start) * s
                y[i] = y_curr

                # reschedule verso nuovo target raw
                live_target = recipe[i] if last_event == "rise" else 0.0
                if abs(live_target - y_end) > resched_eps:
                    rem = max(0.05, t1 - t)
                    y_start = y_curr
                    y_end   = live_target
                    t0 = t
                    t1 = t + rem
        else:
            # tracking live senza limiti
            y[i] = recipe[i]

    return y

# ---- Helper per incroci con soglie
def _find_level_crossings(t: np.ndarray, y: np.ndarray, level: float, atol: float = 1e-6) -> List[float]:
    times: List[float] = []
    for i in range(1, len(t)):
        y0, y1 = y[i-1], y[i]
        t0, t1 = t[i-1], t[i]

        if abs(y1 - level) <= atol and (not times or abs(t1 - times[-1]) > 1e-9):
            times.append(float(t1))
            continue
        if abs(y0 - level) <= atol and (not times or abs(t0 - times[-1]) > 1e-9):
            times.append(float(t0))
            continue

        s0 = y0 - level
        s1 = y1 - level
        if s0 == 0.0 and s1 == 0.0:
            continue
        if s0 * s1 < 0.0:
            alpha = (level - y0) / (y1 - y0)
            tc = t0 + alpha * (t1 - t0)
            if not times or abs(tc - times[-1]) > 1e-9:
                times.append(float(tc))
    return sorted(times)

def _next_after(times: List[float], t_ref: float) -> Optional[float]:
    if t_ref is None:
        return times[0] if times else None
    for x in times:
        if x > t_ref:
            return x
    return None

def draw_time_markers(ax, times_list):
    """
    Disegna linee verticali per i tempi (valori non-None) e scrive la label.
    'times_list' è una lista di tuple (label_str, x_time).
    Il testo è ancorato all'asse X (in alto), così non dipende dalle y-lim.
    """
    used_x = {}
    x_min, x_max = ax.get_xlim()

    for label, x in times_list:
        if x is None:
            continue

        key = round(x, 3)
        dx_count = used_x.get(key, 0)
        used_x[key] = dx_count + 1
        x_plot = x + 0.01 * dx_count

        eps = 0.002 * (x_max - x_min) if x_max > x_min else 0.01
        x_plot = min(max(x_plot, x_min + eps), x_max - eps)

        ax.axvline(x_plot, linestyle=MARKER_LINESTYLE, color=MARKER_COLOR, alpha=MARKER_ALPHA)
        # ax.text(
        #     x_plot, 1.0, label,
        #     transform=ax.get_xaxis_transform(),
        #     ha="center", va="bottom", rotation=90,
        #     fontsize=LABEL_FONTSIZE, color=MARKER_COLOR
        # )



def main():
    csv_path = os.path.join(BAG_DIR, CSV_NAME)
    out_path = os.path.join(BAG_DIR, OUT_NAME)

    if os.path.isfile(csv_path):
        # path classico via CSV
        df = load_signals(csv_path)
        try:
            t = index_to_seconds(df)
        except Exception as e:
            print(f"[ERRORE] Problema con l'indice tempo (CSV): {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # niente CSV → leggo direttamente la rosbag
        print(f"[INFO] CSV non trovato, leggo la bag: {BAG_DIR}")
        df = load_joint0_from_bag(BAG_DIR, topic_name="/joint_states", joint_index=0)
        # l’indice è già in secondi relativi (t=0 alla 1ª misura)
        t = df.index.to_numpy(dtype=float)


    col_pos      = "joint_0_position"    # theta_w
    col_ass_true = "tau_ass_total"       # τ_ass vera (se serve altrove)
    col_mea      = "tau_meas_0"          # se serve altrove

    missing = [c for c in [col_pos] if c not in df.columns]
    if missing:
        print(f"[ATTENZIONE] Mancano colonne nel CSV: {missing}")

    try:
        t_rel, df_win = slice_time_window(t, df, START_S, END_S)
    except Exception as e:
        print(f"[ERRORE] {e}", file=sys.stderr)
        sys.exit(1)

    theta_w = df_win[col_pos].to_numpy() if col_pos in df_win.columns else None

    # --- Rescaling affine di theta_w (ancorato al minimo) ---
    theta_w_orig = theta_w
    theta_w_used = theta_w
    if theta_w is not None:
        th_min = float(np.nanmin(theta_w))
        th_max = float(np.nanmax(theta_w))
        ok_range = (th_max > th_min + 1e-9)
        ok_target = True
        if not ALLOW_EXPANSION and THETA_MAX_TARGET > th_max:
            ok_target = False

        if ok_range and ok_target:
            s = (THETA_MAX_TARGET - th_min) / (th_max - th_min)
            theta_w_used = th_min + s * (theta_w - th_min)

    theta_shift_rad = math.radians(SHIFT_THETA_DEG)
    theta_w_used = theta_w_used + theta_shift_rad

    # Modello da theta_w
    tau_w_model  = compute_tau_w(theta_w_used)  if theta_w is not None else None
    tau_b_model  = compute_tau_box(theta_w_used) if theta_w is not None else None

    # --- Calcolo tempi da soglie orizzontali (calcoli in radianti) ---
    if theta_w_used is None:
        print("[ERRORE] theta_w non disponibile nel CSV.", file=sys.stderr)
        sys.exit(1)

    theta_stand_rad = math.radians(THETA_STAND_DEG)
    theta_bend_rad  = math.radians(THETA_BEND_DEG)

    stand_hits = _find_level_crossings(t_rel, theta_w_used, theta_stand_rad)
    bend_hits  = _find_level_crossings(t_rel, theta_w_used, theta_bend_rad)

    def _require(name: str, val):
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            print(f"[ATTENZIONE] Non trovato tempo per {name}", file=sys.stderr)
        return val

    global T01, T10, T1w, T2, T3, T4, T5w, T6
    T01 = _require("T01 (theta == stand, primo crossing)", stand_hits[0] if len(stand_hits)>0 else None)
    T10 = _require("T10 (theta == bend, primo crossing)",  bend_hits[0]  if len(bend_hits)>0  else None)

    T1w = _require("T1w (theta == bend dopo T1)",  _next_after(bend_hits,  T1))
    T2  = _require("T2  (theta == stand dopo T1w)", _next_after(stand_hits, T1w if T1w is not None else T1))

    T3  = _require("T3  (theta == stand dopo T2)",  _next_after(stand_hits, T2 if T2 is not None else (T1w or T1 or 0.0)))
    T4  = _require("T4  (theta == bend dopo T3)",   _next_after(bend_hits,  T3 if T3 is not None else (T2 or T1 or 0.0)))

    T5w = _require("T5w (theta == bend dopo T5)",   _next_after(bend_hits,  T5))
    T6  = _require("T6  (theta == stand dopo T5w)", _next_after(stand_hits, T5w if T5w is not None else T5))

    # --- Gate temporale per la ricetta no_vision: attiva solo in [T1w,T2], [T3,T4], [T5w,T6]
    gate_novis = np.zeros_like(t_rel, dtype=float)

    def seg_mask(a, b):
        if a is None or b is None:
            return np.zeros_like(t_rel, dtype=bool)
        return (t_rel >= a) & (t_rel < b)

    gate_novis[seg_mask(T1w, T2)] = 1.0    # prima finestra spostata da T1 -> T1w
    gate_novis[seg_mask(T3,  T4)] = 1.0    # invariata
    # includiamo anche il punto T6 come già fatto altrove
    if (T5w is not None) and (T6 is not None):
        mask_56 = (t_rel >= T5w) & (t_rel <= T6)
        gate_novis[mask_56] = 1.0


    # Pesi schedule per ciascun istante t_rel (robusto ai None)
    w_w, w_b = schedule_weights(t_rel, full_box_seg34=True)

    # Ricette (PRIMA delle rampe) **SENZA LIMITE**
    tau_ass_withbox_recipe = None
    tau_ass_nobox_recipe   = None
    if (tau_w_model is not None) and (tau_b_model is not None):
        raw_withbox = w_w * tau_w_model + w_b * tau_b_model
        raw_nobox   = w_w * tau_w_model + 0.0   # senza box

        tau_ass_withbox_recipe = coeff_assist * raw_withbox
        tau_ass_nobox_recipe   = coeff_assist * raw_nobox * gate_novis  # <<--- GATE APPLICATO

    # Eventi rampe:
    # with_vision: RISES = [T1, T3, T5], FALLS = [T2, T4, T6]
    # no_vision : RISES = [T1w, T3, T5w], FALLS = [T2, T4, T6]
    RISES_WITH  = [T1, T3, T5]
    FALLS_WITH  = [T2, T4, T6]

    # _r_T1w = T1w if (T1w is not None) else T1
    # _r_T5w = T5w if (T5w is not None) else T5
    RISES_NOVIS = [T1w, T3, T5w]
    FALLS_NOVIS = [T2, T4, T6]

    # Rampe → "desired" (interne), poi clamp → "applied" (da plottare)
    tau_ass_withbox_des = apply_event_ramps(
        tau_ass_withbox_recipe, t_rel, RISES_WITH, FALLS_WITH, RAMP_S, resched_eps=RAMP_EPS
    ) if tau_ass_withbox_recipe is not None else None
    tau_ass_nobox_des   = apply_event_ramps(
        tau_ass_nobox_recipe,   t_rel, RISES_NOVIS, FALLS_NOVIS, RAMP_S, resched_eps=RAMP_EPS
    ) if tau_ass_nobox_recipe is not None else None

    tau_ass_withbox_appl = clamp(tau_ass_withbox_des, -assist_max_nm, assist_max_nm) if tau_ass_withbox_des is not None else None
    tau_ass_nobox_appl   = clamp(tau_ass_nobox_des,   -assist_max_nm, assist_max_nm) if tau_ass_nobox_des is not None else None

    # === PLOT (3 pannelli) ===

    TIMES = [
        (r"$T_{0,\mathrm{in}}$",                      T0),   # T0  -> T_{0,in}
        (r"$T_{0,\mathrm{end}}$",                     T01),  # T01 -> T_{0,end}
        (r"$T_{1,\mathrm{in}}$",                      T10),  # T10 -> T_{1,in}
        (r"$T_{1,\mathrm{end}}^{\mathrm{vis}}$",      T1),   # T1  -> T_{1,end}^{vis}
        (r"$T_{1,\mathrm{end}}^{\mathrm{novis}}$",    T1w),  # T1w -> T_{1,end}^{novis}
        (r"$T_{2,\mathrm{in}}$",                      T2),   # T2  -> T_{2,in}
        (r"$T_{2,\mathrm{end}}$",                     T3),   # T3  -> T_{2,end}
        (r"$T_{3,\mathrm{in}}$",                      T4),   # T4  -> T_{3,in}
        (r"$T_{3,\mathrm{end}}^{\mathrm{vis}}$",      T5),   # T5  -> T_{3,end}^{vis}
        (r"$T_{3,\mathrm{end}}^{\mathrm{novis}}$",    T5w),  # T5w -> T_{3,end}^{novis}
        (r"$T_{0,\mathrm{in}}^{\mathrm{new}}$",       T6),   # T6  -> T_{0,in}^{new}
    ]


    n_rows = 3
    # plt.figure(figsize=(12, 9))
    fig = plt.figure(figsize=(12, 9), constrained_layout=True)


    # (1) Posizione (in gradi opzionale)
    ax1 = plt.subplot(n_rows, 1, 1)
    if theta_w_used is not None:
        if SHOW_POS_IN_DEG:
            theta_plot = np.degrees(theta_w_used)
            ax1.plot(t_rel, theta_plot, color=POS_COLOR, label=r"$\theta_w$")
        else:
            ax1.plot(t_rel, theta_w_used, color=POS_COLOR, label=r"$\theta_w$")
    if SHOW_ORIG_THETA and (theta_w_orig is not None):
        if SHOW_POS_IN_DEG:
            theta_orig_plot = np.degrees(theta_w_orig)
            ax1.plot(t_rel, theta_orig_plot, linestyle=":", label="theta_w (orig)")
        else:
            ax1.plot(t_rel, theta_w_orig, linestyle=":", label="theta_w (orig)")

    # Linee orizzontali soglie
    if SHOW_THETA_BEND:
        if SHOW_POS_IN_DEG:
            ax1.axhline(THETA_BEND_DEG, color="k", linestyle="-", linewidth=1.5,
                        label=rf"$\theta_{{\mathrm{{bend}}}}={THETA_BEND_DEG:.0f}^\circ$")
        else:
            ax1.axhline(math.radians(THETA_BEND_DEG), color="k", linestyle="-", linewidth=1.5,
                        label=rf"$\theta_{{\mathrm{{bend}}}}={THETA_BEND_DEG:.0f}^\circ$")
    
    if SHOW_THETA_STAND:
        if SHOW_POS_IN_DEG:
            ax1.axhline(THETA_STAND_DEG, color="k", linestyle="--", linewidth=1.5,
                        label=rf"$\theta_{{\mathrm{{stand}}}}={THETA_STAND_DEG:.0f}^\circ$")
        else:
            ax1.axhline(math.radians(THETA_STAND_DEG), color="k", linestyle=":", linewidth=1.5,
                        label=rf"$\theta_{{\mathrm{{stand}}}}={THETA_STAND_DEG:.0f}^\circ$")

    # # Marker verticali
    # for x in [T01, T10, T1, T1w, T2, T3, T4, T5, T5w, T6]:
    #     if x is not None:
    #         ax1.axvline(x, linestyle="--", alpha=0.5)

    ax1.set_xlabel("t [s]", loc="right")
    ax1.set_ylabel(r"$\theta$ [deg]" if SHOW_POS_IN_DEG else "theta_w [rad]")
    ax1.set_title("Lumbar Position")
    ax1.grid(True)
    ax1.legend(loc="best")
    if SHOW_POS_IN_DEG:
        ax1.yaxis.set_major_locator(MultipleLocator(10))
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    
    # draw_time_markers(ax1, TIMES)

    # (2) Modello: tau_w & tau_box
    ax2 = plt.subplot(n_rows, 1, 2, sharex=ax1)
    if tau_w_model is not None:
        ax2.plot(t_rel, tau_w_model, color=TAU_W_COLOR, label=r"$\tau_w$")
    if tau_b_model is not None:
        ax2.plot(t_rel, tau_b_model, color=TAU_BOX_COLOR, label=r"$\tau_{\mathrm{box}}$")
    # for x in [T01, T10, T1, T1w, T2, T3, T4, T5, T5w, T6]:
    #     if x is not None:
    #         ax2.axvline(x, linestyle="--", alpha=0.3)
    ax2.set_xlabel("t [s]", loc="right")
    ax2.set_ylabel("Torque [Nm]")
    ax2.set_title("Weight and Box torques")
    ax2.grid(True)
    ax2.legend(loc="best")

    ax2.yaxis.set_major_locator(MultipleLocator(40 if 'TORQUE_TICK_NM' in globals() else 10))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    # draw_time_markers(ax2, TIMES)

    # (3) tau_ass — SOLO applied (no desired)
    ax3 = plt.subplot(n_rows, 1, 3, sharex=ax1)
    # if tau_ass_nobox_appl is not None:
    #     ax3.plot(t_rel, tau_ass_nobox_appl,
    #              label=r"$\tau_{\mathrm{ass,\,novis}}$")
    if tau_ass_withbox_appl is not None:
        ax3.plot(t_rel, tau_ass_withbox_appl,
                 label=r"$\tau_{\mathrm{ass,\,vis}}$")
    # for x in [T01, T10, T1, T1w, T2, T3, T4, T5, T5w, T6]:
    #     if x is not None:
    #         ax3.axvline(x, linestyle="--", alpha=0.3)
    ax3.set_xlabel("t [s]", loc="right")
    ax3.set_ylabel("Torque [Nm]")
    ax3.set_title("Total assistance torque")
    ax3.grid(True)
    ax3.legend(loc="best")
    ax3.xaxis.set_major_locator(MultipleLocator(TIME_TICK_S))
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax3.yaxis.set_major_locator(MultipleLocator(5))
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    # ---- ANIMAZIONE in tempo reale sulla finestra (durata = END_S - START_S) ----
    from matplotlib.animation import FuncAnimation

    ENABLE_ANIMATION = True
    DOT_SIZE = 60

    # palline (solo ax1 e ax3, niente linee verticali)
    ball1, = ax1.plot([], [], marker='o', markersize=8, linestyle='None', color='tab:red')
    ball3, = ax3.plot([], [], marker='o', markersize=8, linestyle='None', color='tab:red')

    y_pos = np.degrees(theta_w_used) if SHOW_POS_IN_DEG else theta_w_used
    y_ass = tau_ass_withbox_appl if tau_ass_withbox_appl is not None else None

    # evita rescale durante l’animazione
    for ax in (ax1, ax2, ax3):
        ax.relim(); ax.autoscale_view()

    def init():
        ball1.set_data([], [])
        ball3.set_data([], [])
        return ball1, ball3

    def update(i):
        t = t_anim[i]
        # posizione
        ball1.set_data([t], [y_pos[i]])
        # tau_ass (no-vision)
        if y_ass is not None:
            ball3.set_data([t], [y_ass[i]])
        return ball1, ball3

    ani = None
    if ENABLE_ANIMATION and len(t_rel) > 1:
        # un frame per campione, durata esatta = t_rel[-1] - t_rel[0] == END_S - START_S
        t_anim = np.linspace(t_rel[0], t_rel[-1], len(t_rel))
        t_span = float(t_anim[-1] - t_anim[0])
        fps = (len(t_anim) - 1) / max(1e-9, t_span)   # così (N-1)/fps = t_span
        interval_ms = 1000.0 / max(1.0, fps*2) #speedfactor

        ani = FuncAnimation(fig, update,
                            frames=len(t_anim),
                            init_func=init,
                            interval=interval_ms,
                            blit=True,
                            repeat=False)
        
            # --- Salvataggio MP4 con lo stesso rate dell'animazione a video ---
        try:
            # usa l'fps effettivo dell'animazione: fps_writer = 1000 / interval_ms
            fps_writer = max(1, int(round(1000.0 / interval_ms)))
            mp4_path = os.path.join(BAG_DIR, OUT_MP4_NAME)

            writer = FFMpegWriter(
                fps=fps_writer,
                codec='libx264',
                bitrate=MP4_BITRATE,
                extra_args=['-pix_fmt', 'yuv420p', '-movflags', '+faststart']
            )
            ani.save(mp4_path, writer=writer, dpi=MP4_DPI)
            print(f"[OK] Video salvato in: {mp4_path} (fps={fps_writer})")
        except KeyboardInterrupt:
            print("[WARN] Salvataggio MP4 interrotto manualmente (Ctrl-C).")
        except Exception as e:
            print(f"[WARN] Impossibile salvare MP4 ({e}). Verifica che 'ffmpeg' sia installato e nel PATH.")




    # draw_time_markers(ax3, TIMES)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[OK] Grafico salvato in: {out_path}")

    if SHOW_FIG:
        plt.show()

if __name__ == "__main__":
    main()
