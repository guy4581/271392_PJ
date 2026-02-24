import cv2
from ultralytics import YOLO

model = YOLO("best (1).pt")
cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # 1. ดึงขนาดของภาพ (Height, Width)
        height, width = frame.shape[:2]
        
        # 2. คำนวณจุดกึ่งกลางของกล้อง (Camera Center)
        cam_cx = width // 2
        
        results = model.track(frame, persist=True, device="cpu")
        annotated_frame = results[0].plot()
        
        # --- วาดเส้นอ้างอิงกึ่งกลางกล้อง (สีเขียว - Target Line) ---
        cv2.line(
            annotated_frame, 
            (cam_cx, 0),        # จุดเริ่มบนสุดกลางจอ
            (cam_cx, height),   # จุดจบด้านล่างกลางจอ
            (0, 255, 0),        # สีเขียว (เป้าหมาย)
            thickness=2
        )
        
        # วาดเส้นของ BBox (ถ้าตรวจจับเจอ)
        if results[0].boxes is not None:
            for box in results[0].boxes:
                # ดึงพิกัด x ของ BBox และแปลงเป็น Integer
                bbox_x = int(box.xywh[0][0].item())
                
                # --- วาดเส้นแนวตั้งของ BBox (สีน้ำเงิน - Current Position) ---
                cv2.line(
                    annotated_frame, 
                    (bbox_x, 0), 
                    (bbox_x, height), 
                    (255, 0, 0), # สีน้ำเงิน
                    thickness=2
                )
                
                # --- คำนวณ Error (ระยะห่างระหว่างเส้นเขียวและน้ำเงิน) ---
                error = bbox_x - cam_cx
                cv2.putText(annotated_frame, f"Error: {error} px", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("YOLO11 Tracking & Center Ref", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()