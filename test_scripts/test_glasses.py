import cv2

# Questo URL funziona con g3pycontroller
RTSP_URL = "rtsp://192.168.75.51:8554/live/all"

cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("❌ Impossibile aprire il flusso RTSP")
    exit()

print("✅ Flusso RTSP aperto con successo")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame non ricevuto")
        break

    cv2.imshow("Tobii Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
