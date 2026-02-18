import os
import time
import asyncio
import cv2
from g3pylib import connect_to_glasses

HOSTNAME = "192.168.75.51"
SAVE_DIR = "/home/alessandro/exo_v2_ws/frames_yolo"
SAVE_INTERVAL_S = 1.0


async def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    last_save = 0.0
    count = 0

    async with connect_to_glasses.with_hostname(HOSTNAME) as g3:
        async with g3.stream_rtsp(scene_camera=True, gaze=False) as streams:
            async with streams.scene_camera.decode() as scene_stream:
                print(f"Saving frames every {SAVE_INTERVAL_S:.1f}s to {SAVE_DIR}")
                while True:
                    frame, ts = await scene_stream.get()
                    if frame is None:
                        continue

                    now = time.time()
                    if now - last_save < SAVE_INTERVAL_S:
                        continue

                    img = frame.to_ndarray(format="bgr24")
                    fname = f"frame_{int(now*1000)}_{count:06d}.jpg"
                    path = os.path.join(SAVE_DIR, fname)
                    ok = cv2.imwrite(path, img)
                    if ok:
                        print(f"[saved] {path}")
                        count += 1
                        last_save = now
                    else:
                        print(f"[error] failed to save {path}")


if __name__ == "__main__":
    asyncio.run(main())
