import cv2
import math
import numpy as np
from ultralytics import YOLO

model = YOLO("best (1).pt")
cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        height, width = frame.shape[:2]
        cam_cx = width // 2  # เส้นอ้างอิงกึ่งกลางกล้อง
        
        results = model.track(frame, persist=True, device="cpu")
        annotated_frame = results[0].plot()
        
        # 1. วาดเส้นอ้างอิงกึ่งกลางกล้อง (เป้าหมาย - สีเขียว)
        cv2.line(annotated_frame, (cam_cx, 0), (cam_cx, height), (0, 255, 0), 2)
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                # ดึงพิกัดขอบกล่อง [x1, y1, x2, y2]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # หาจุดกึ่งกลางด้านบนและด้านล่างของ BBox แปลง
                top_center_x = (x1 + x2) / 2
                bottom_center_x = (x1 + x2) / 2 # กรณีกล่องตั้งตรง ค่า x จะเท่ากัน
                
                # ในกรณีที่ต้องการวัดความเอียงจากรูปทรงกล่อง (ที่อาจจะเอียงตาม Perspective)
                # เราจะคำนวณมุมจากจุดกึ่งกลาง BBox (cx, cy) เทียบกับจุดอ้างอิง
                cx, cy, w, h = box.xywh[0].tolist()

                # --- วิธีคำนวณ Heading Error (มุมเอียง) ---
                # เราจะใช้ค่าความต่างของ x ที่จุดบนและล่างของกล่อง (ถ้าโมเดลเป็น Segment จะแม่นกว่า)
                # แต่สำหรับ BBox ทั่วไป เราจะวัดจากจุดกึ่งกลางภาพไปยังกึ่งกลางกล่อง
                dx = cx - cam_cx
                dy = height - cy # ระยะห่างจากด้านล่างจอ
                
                angle_error = math.degrees(math.atan2(dx, dy))

                # 2. วาดเส้นแนวกลางของแปลงที่ Detect เจอ (สีน้ำเงิน)
                cv2.line(annotated_frame, (int(cx), 0), (int(cx), height), (255, 0, 0), 2)

                # แสดงค่า Error
                cv2.putText(annotated_frame, f"Heading Error: {angle_error:.2f} deg", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # ระยะห่างจากกึ่งกลาง (Lateral Error)
                lateral_error = cx - cam_cx
                cv2.putText(annotated_frame, f"Lat Error: {int(lateral_error)} px", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Empty Field Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break
    else: break

cap.release()
cv2.destroyAllWindows()