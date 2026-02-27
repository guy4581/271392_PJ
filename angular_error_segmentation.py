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
# scan line บน / ล่าง
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
    # segmentation
    # =========================
    results = model.track(frame, persist=True, device=0)

    mask_img = np.zeros((h, w), dtype=np.uint8)

    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()

        for m in masks:
            m = (m * 255).astype(np.uint8)
            m = cv2.resize(m, (w, h))
            mask_img = cv2.bitwise_or(mask_img, m)

    vis = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)

    # =========================
    # เส้น reference กล้อง
    # =========================
    cv2.line(vis, (cam_center, 0), (cam_center, h), (255, 0, 0), 2)

    # scan lines
    cv2.line(vis, (0, y_top), (w, y_top), (255,255,0), 2)
    cv2.line(vis, (0, y_bottom), (w, y_bottom), (255,255,0), 2)

    # =========================
    # ฟังก์ชันหา center แปลงในแถว
    # =========================
    def get_lane_center(y):
        row = mask_img[y]
        xs = np.where(row > 0)[0]
        if len(xs) == 0:
            return None
        return (xs.min() + xs.max()) // 2

    cx_top = get_lane_center(y_top)
    cx_bottom = get_lane_center(y_bottom)

    # =========================
    # ถ้าเจอทั้งบนและล่าง
    # =========================
    if cx_top is not None and cx_bottom is not None:

        # วาด center points
        cv2.circle(vis, (cx_top, y_top), 8, (0,0,255), -1)
        cv2.circle(vis, (cx_bottom, y_bottom), 8, (0,0,255), -1)

        # วาดเส้นแนวแปลง
        cv2.line(vis, (cx_top, y_top), (cx_bottom, y_bottom), (0,255,0), 3)

        # =========================
        # LATERAL ERROR
        # =========================
        lateral_error = cx_bottom - cam_center

        # =========================
        # HEADING ERROR (มุมเอียง)
        # =========================
        dx = cx_top - cx_bottom
        dy = y_top - y_bottom

        angle_rad = math.atan2(dx, -dy)
        angle_deg = math.degrees(angle_rad)

        # =========================
        # แสดงผล
        # =========================
        cv2.putText(vis, f"Lateral: {lateral_error}px",
                    (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        cv2.putText(vis, f"Heading: {angle_deg:.2f} deg",
                    (30,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        print("Lateral:", lateral_error, " Heading:", angle_deg)

    else:
        cv2.putText(vis, "LANE NOT DETECTED",
                    (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Bird Eye Alignment", vis)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()