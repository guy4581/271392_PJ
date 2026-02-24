import cv2
import math
from ultralytics import YOLO

model = YOLO("best (1).pt")
cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        height, width = frame.shape[:2]
        cam_cx = width // 2
        
        # จุดอ้างอิงกึ่งกลางล่างภาพ (Pivot Point)
        pivot_point = (cam_cx, height) 
        
        results = model.track(frame, persist=True, device="cpu")
        annotated_frame = results[0].plot()
        
        # 1. วาดเส้นอ้างอิงกึ่งกลางกล้อง (เป้าหมาย - สีเขียว)
        cv2.line(annotated_frame, (cam_cx, 0), (cam_cx, height), (0, 255, 0), 2)
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                # ดึงพิกัด x, y กึ่งกลาง BBox
                bbox_x = int(box.xywh[0][0].item())
                bbox_y = int(box.xywh[0][1].item())
                
                # 2. วาดเส้นจากจุด Pivot ไปยังกลาง BBox (สีน้ำเงิน)
                cv2.line(annotated_frame, pivot_point, (bbox_x, bbox_y), (255, 0, 0), 2)
                
                # 3. คำนวณมุม (Angle Error)
                # dx = ระยะห่างแนวแกน X, dy = ระยะห่างแนวแกน Y จากจุดล่างสุด
                dx = bbox_x - cam_cx
                dy = height - bbox_y  # ระยะจากขอบล่างขึ้นไปถึงจุดกลาง BBox
                
                # ใช้ atan2 เพื่อหามุมเป็นเรเดียน และแปลงเป็นองศา
                # มุมที่ได้จะเป็น 0 เมื่อ dx = 0 (ทับเส้นสีเขียวพอดี)
                angle_error = math.degrees(math.atan2(dx, dy))
                
                # แสดงค่า Error เป็นองศา
                cv2.putText(annotated_frame, f"Angle Error: {angle_error:.2f} deg", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # วาดเส้นแนวตั้งของ BBox เดิมไว้เปรียบเทียบ (สีน้ำเงินจาง)
                cv2.line(annotated_frame, (bbox_x, 0), (bbox_x, height), (255, 0, 0), 1)

        cv2.imshow("Angle Error Calculation", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()