import os
import threading
import asyncio
import time
import cv2
from g3pylib import connect_to_glasses

HOSTNAME  = "192.168.75.51"
RTSP_URL  = "rtsp://192.168.75.51:8554/live/all"   # prova anche .../live/scene se serve

# Forza FFmpeg a usare TCP e timeouts più lunghi (microsecondi per stimeout)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;10000000|timeout;10000000"

latest_gaze_xy = None
stop_flag = False

def extract_gaze_xy(gf):
    if not gf or not isinstance(gf, dict):
        return None
    if "gaze2d" in gf and gf["gaze2d"] is not None:
        return gf["gaze2d"]
    if "data" in gf and isinstance(gf["data"], dict):
        g = gf["data"].get("gaze2d")
        if g is not None:
            return g
    return None

async def gaze_and_start_rtsp_worker():
    """
    Avvia *ufficialmente* la sessione RTSP lato occhiali (scene_camera=True)
    e legge SOLO il gaze. Non decodifichiamo il video qui; lo prenderà OpenCV via RTSP_URL.
    """
    global latest_gaze_xy, stop_flag
    last_log = 0
    async with connect_to_glasses.with_hostname(HOSTNAME) as g3:
        # QUI la differenza: scene_camera=True per far partire l'RTSP server
        async with g3.stream_rtsp(scene_camera=True, gaze=True) as streams:
            # Non decodificare la scena qui: basta tener viva la sessione RTSP.
            async with streams.gaze.decode() as gaze_stream:
                print("[gaze] connesso. Sessione RTSP avviata (scene_camera=True).")
                while not stop_flag:
                    try:
                        gaze_frame, ts = await gaze_stream.get()
                        xy = extract_gaze_xy(gaze_frame)
                        if xy is not None:
                            latest_gaze_xy = xy
                            print(f"Gaze: x={xy[0]:.4f}, y={xy[1]:.4f}")
                        else:
                            now = time.time()
                            if now - last_log > 2:
                                print("[gaze] nessun dato (indossa/calibra, privacy OFF).")
                                last_log = now
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        print(f"[gaze] errore: {e}")
                        await asyncio.sleep(0.05)

def start_gaze_thread():
    loop = asyncio.new_event_loop()
    def runner():
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(gaze_and_start_rtsp_worker())
        finally:
            loop.close()
    t = threading.Thread(target=runner, daemon=True)
    t.start()
    return loop, t

def main():
    global stop_flag
    # 1) Avvia worker che accende RTSP e legge gaze
    loop, t = start_gaze_thread()

    # 2) Apri RTSP con OpenCV (usa CAP_FFMPEG esplicitamente)
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("❌ Impossibile aprire il flusso RTSP (OpenCV)")
        stop_flag = True
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=1.0)
        return

    # (opzionale) riduci buffer per latenza
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    print("✅ RTSP aperto. Premi 'q' per uscire.")
    cv2.namedWindow("Tobii Stream + Gaze", cv2.WINDOW_NORMAL)

    idle_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            idle_count += 1
            if idle_count % 50 == 0:
                print("⚠️ Nessun frame RTSP (controllo connessione/RTSP attivo).")
            # piccolo sleep per non saturare CPU
            time.sleep(0.01)
            # continua a tentare: la sessione RTSP potrebbe attivarsi con qualche secondo di ritardo
            continue
        idle_count = 0

        xy = latest_gaze_xy
        if xy is not None:
            h, w = frame.shape[:2]
            cx = int(xy[0] * w)
            cy = int((1.0 - xy[1]) * h)
            cv2.circle(frame, (cx, cy), 8, (0, 0, 255), 2)

        cv2.imshow("Tobii Stream + Gaze", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    stop_flag = True
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=1.0)

if __name__ == "__main__":
    main()
