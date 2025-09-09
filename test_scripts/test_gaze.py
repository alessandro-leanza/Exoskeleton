import asyncio
import time
import threading
import cv2
import numpy as np
from g3pylib import connect_to_glasses

HOSTNAME = "192.168.75.51"  # IP occhiali

latest_frame = None       # np.ndarray BGR
latest_gaze_xy = None     # [x, y] in [0..1]
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

async def reader_task_scene(scene_stream):
    global latest_frame, stop_flag
    while not stop_flag:
        try:
            frame, ts = await scene_stream.get()
            if frame is None:
                continue
            # PyAV -> numpy BGR
            img = frame.to_ndarray(format="bgr24")
            latest_frame = img
        except Exception as e:
            print(f"[scene] errore: {e}")
            await asyncio.sleep(0.05)

async def reader_task_gaze(gaze_stream):
    global latest_gaze_xy, stop_flag
    last_log = 0
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
                    print("Nessun dato gaze (indossa/calibra occhiali, privacy OFF).")
                    last_log = now
        except Exception as e:
            print(f"[gaze] errore: {e}")
            await asyncio.sleep(0.05)

def render_loop():
    global stop_flag, latest_frame, latest_gaze_xy
    # Prova a creare una finestra: se fallisce, probabilmente OpenCV è headless
    try:
        cv2.namedWindow("Tobii G3 - Scene + Gaze", cv2.WINDOW_NORMAL)
    except Exception as e:
        print("Impossibile aprire finestra OpenCV. OpenCV è headless o manca DISPLAY.")
        print("Suggerimenti: installa GUI backend (es. libgtk-3-dev) e ricompila/installa opencv-python con GTK/Qt; oppure usa un ambiente non-SSH.")
        stop_flag = True
        return

    while not stop_flag:
        img = latest_frame
        if img is not None:
            img_draw = img.copy()
            h, w = img_draw.shape[:2]
            xy = latest_gaze_xy
            if xy is not None:
                cx = int(xy[0] * w)
                cy = int((1.0 - xy[1]) * h)  # inverti Y
                cv2.circle(img_draw, (cx, cy), 8, (0, 0, 255), 2)
            cv2.imshow("Tobii G3 - Scene + Gaze", img_draw)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_flag = True
                break
        else:
            # nessun frame ancora, aspetta poco
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_flag = True
                break
            time.sleep(0.01)

    cv2.destroyAllWindows()

async def main():
    global stop_flag
    async with connect_to_glasses.with_hostname(HOSTNAME) as g3:
        async with g3.stream_rtsp(scene_camera=True, gaze=True) as streams:
            async with streams.scene_camera.decode() as scene_stream, streams.gaze.decode() as gaze_stream:
                print("Connesso. Premi 'q' nella finestra video per uscire.")
                # Avvia thread per rendering OpenCV (evita blocchi con asyncio)
                t = threading.Thread(target=render_loop, daemon=True)
                t.start()

                # Avvia i reader async
                task_scene = asyncio.create_task(reader_task_scene(scene_stream))
                task_gaze = asyncio.create_task(reader_task_gaze(gaze_stream))

                # Attendi finché non si chiude
                try:
                    while not stop_flag:
                        await asyncio.sleep(0.05)
                finally:
                    stop_flag = True
                    task_scene.cancel()
                    task_gaze.cancel()
                    try:
                        await task_scene
                    except asyncio.CancelledError:
                        pass
                    try:
                        await task_gaze
                    except asyncio.CancelledError:
                        pass

if __name__ == "__main__":
    asyncio.run(main())
