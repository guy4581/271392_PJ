import cv2
import numpy as np
from ultralytics import YOLO
import math

# =========================
# โหลดโมเดล
# =========================
model = YOLO("rack_segm.pt")

cap = cv2.VideoCapture(1)

# =========================
# scan line บน / ล่าง (เส้นอ้างอิงกล้อง)
# =========================
TOP_RATIO = 0.25
BOTTOM_RATIO = 0.70

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w = frame.shape[:2]
    cam_center = w // 2

    y_top = int(h * TOP_RATIO)
    y_bottom = int(h * BOTTOM_RATIO)

    # =========================
    # Segmentation
    # =========================
    results = model.track(frame, persist=True, device=0)

    mask_img = np.zeros((h, w), dtype=np.uint8)

    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()

        for m in masks:
            m = (m * 255).astype(np.uint8)
            m = cv2.resize(m, (w, h))
            mask_img = cv2.bitwise_or(mask_img, m)

    # แปลงภาพขาวดำเป็นสีเพื่อวาดข้อมูลทับ
    vis = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)

    # =========================
    # วาดเส้น reference กล้อง
    # =========================
    cv2.line(vis, (cam_center, 0), (cam_center, h), (255, 0, 0), 2)
    cv2.line(vis, (0, y_top), (w, y_top), (255, 255, 0), 2)
    cv2.line(vis, (0, y_bottom), (w, y_bottom), (255, 255, 0), 2)

    # =========================
    # 1. หา Contour และจุดบนสุด/ล่างสุดของ Object
    # =========================
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # เลือก Contour ที่ใหญ่ที่สุด (ป้องกัน noise)
        c = max(contours, key=cv2.contourArea)
        
        # วาดเส้นขอบ Contour ของวัตถุ
        cv2.drawContours(vis, [c], -1, (255, 0, 255), 2)

        # หาจุด Extreme Top และ Extreme Bottom จาก Contour
        # c[:, :, 1] คือค่าแกน Y ของทุกจุดใน contour
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        # วาดจุดบนสุดและล่างสุดของ Object
        cv2.circle(vis, extTop, 8, (0, 165, 255), -1) # สีส้ม
        cv2.circle(vis, extBot, 8, (0, 165, 255), -1)

        # คำนวณระยะห่าง (แกน Y) ระหว่างขอบวัตถุ กับ เส้นอ้างอิงกล้อง
        # ค่าบวกแปลว่า ขอบวัตถุอยู่ "ต่ำกว่า" เส้นอ้างอิง, ค่าลบแปลว่า "สูงกว่า" (ตามแกน Y ของภาพที่บนสุดคือ 0)
        dist_top = extTop[1] - y_top
        dist_bottom = extBot[1] - y_bottom

        # วาดเส้นแสดงระยะห่าง
        cv2.line(vis, extTop, (extTop[0], y_top), (255, 255, 255), 2)
        cv2.line(vis, extBot, (extBot[0], y_bottom), (255, 255, 255), 2)

        # แสดงผลตัวเลขระยะห่างบนจอ
        cv2.putText(vis, f"Dist to Top Line: {dist_top}px",
                    (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
        cv2.putText(vis, f"Dist to Bot Line: {dist_bottom}px",
                    (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
    else:
        cv2.putText(vis, "OBJECT NOT DETECTED",
                    (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # =========================
    # 2. ฟังก์ชันหา Center ของเลน/แปลง เพื่อหา Heading (โค้ดเดิมของคุณ)
    # =========================
    def get_lane_center(y):
        row = mask_img[y]
        xs = np.where(row > 0)[0]
        if len(xs) == 0:
            return None
        return (xs.min() + xs.max()) // 2

    cx_top = get_lane_center(y_top)
    cx_bottom = get_lane_center(y_bottom)

    if cx_top is not None and cx_bottom is not None:
        cv2.circle(vis, (cx_top, y_top), 8, (0, 0, 255), -1)
        cv2.circle(vis, (cx_bottom, y_bottom), 8, (0, 0, 255), -1)
        cv2.line(vis, (cx_top, y_top), (cx_bottom, y_bottom), (0, 255, 0), 3)

        lateral_error = cx_bottom - cam_center
        dx = cx_top - cx_bottom
        dy = y_top - y_bottom
        angle_rad = math.atan2(dx, -dy)
        angle_deg = math.degrees(angle_rad)

        cv2.putText(vis, f"Lateral: {lateral_error}px",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(vis, f"Heading: {angle_deg:.2f} deg",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        cv2.putText(vis, "LANE NOT CROSSED",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # =========================
    # แสดงผล
    # =========================
    cv2.imshow("Bird Eye Alignment", vis)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()