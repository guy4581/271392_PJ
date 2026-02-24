import cv2
import math
import numpy as np
from ultralytics import YOLO

model = YOLO("rack.pt")
cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        height, width = frame.shape[:2]
        cam_cx = width // 2
        
        results = model.track(frame, persist=True, device="cpu", conf=0.8)
        annotated_frame = results[0].plot()
        
        cv2.line(annotated_frame, (cam_cx, 0), (cam_cx, height), (0, 255, 0), 2)
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                # พิกัด bbox
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # center บน และ ล่าง ของ bbox
                top_center = (int((x1 + x2) / 2), int(y1))
                bottom_center = (int((x1 + x2) / 2), int(y2))

                # วาดแค่จุด (วงกลม)
                cv2.circle(annotated_frame, top_center, 6, (0, 0, 255), -1)      # บน = แดง
                cv2.circle(annotated_frame, bottom_center, 6, (255, 0, 0), -1)   # ล่าง = น้ำเงิน

                # center bbox
                cx, cy, w, h = box.xywh[0].tolist()

                dx = cx - cam_cx
                dy = height - cy
                angle_error = math.degrees(math.atan2(dx, dy))

                lateral_error = cx - cam_cx

                cv2.putText(annotated_frame, f"Heading Error: {angle_error:.2f} deg", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.putText(annotated_frame, f"Lat Error: {int(lateral_error)} px", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Empty Field Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
