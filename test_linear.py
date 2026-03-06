import cv2
import math
from ultralytics import YOLO

model = YOLO("rack.pt")

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    height, width = frame.shape[:2]
    cam_cx = width // 2

    # ใช้ detect แทน track
    results = model(frame, device="cpu", conf=0.8)

    annotated_frame = results[0].plot()

    cv2.line(annotated_frame, (cam_cx, 0), (cam_cx, height), (0, 255, 0), 2)

    cv2.imshow("Fast YOLO", annotated_frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()