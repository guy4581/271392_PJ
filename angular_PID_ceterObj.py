import cv2
import numpy as np
from ultralytics import YOLO

# ===============================
# LOAD MODEL + CAMERA
# ===============================
model = YOLO("rack_segm.pt")
cap = cv2.VideoCapture(1)

# ===============================
# KALMAN FILTER INIT
# ===============================
kalman = cv2.KalmanFilter(4, 2)

# state = [x, y, vx, vy]
kalman.transitionMatrix = np.array([
    [1,0,1,0],
    [0,1,0,1],
    [0,0,1,0],
    [0,0,0,1]
], np.float32)

kalman.measurementMatrix = np.array([
    [1,0,0,0],
    [0,1,0,0]
], np.float32)

# ปรับความนิ่งตรงนี้
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
kalman.errorCovPost = np.eye(4, dtype=np.float32)

kalman_initialized = False


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w = frame.shape[:2]

    # ===============================
    # RUN TRACKING
    # ===============================
    results = model.track(frame, persist=True, device=0)

    # ===============================
    # CREATE MASK FROM SEGMENTATION
    # ===============================
    mask_img = np.zeros((h, w), dtype=np.uint8)

    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        for m in masks:
            m = (m * 255).astype(np.uint8)
            m = cv2.resize(m, (w, h))
            mask_img = cv2.bitwise_or(mask_img, m)

    # ===============================
    # NOISE REDUCTION
    # ===============================
    mask_img = cv2.medianBlur(mask_img, 5)

    kernel = np.ones((9,9), np.uint8)
    mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel)
    mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel)

    # ===============================
    # KEEP ONLY LARGEST OBJECT
    # ===============================
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    clean_mask = np.zeros_like(mask_img)
    if contours:
        c_big = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean_mask, [c_big], -1, 255, -1)

    mask_img = clean_mask

    # ===============================
    # DISPLAY
    # ===============================
    display = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)

    cam_center_y = h // 2
    cv2.line(display, (0, cam_center_y), (w, cam_center_y), (0,255,0), 2)

    # ===============================
    # FIND OBJECT
    # ===============================
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = False

    if len(contours) > 0:
        detected = True
        c = max(contours, key=cv2.contourArea)

        # contour
        cv2.drawContours(display, [c], -1, (0,255,255), 2)

        # rotated box
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(display, [box], 0, (255,255,0), 2)

        # RAW CENTER (measurement)
        cx_raw = int(rect[0][0])
        cy_raw = int(rect[0][1])

        # ===============================
        # INIT KALMAN
        # ===============================
        if not kalman_initialized:
            kalman.statePost = np.array([[cx_raw],[cy_raw],[0],[0]], np.float32)
            kalman_initialized = True

        # ===============================
        # KALMAN PREDICT + CORRECT
        # ===============================
        kalman.predict()

        measurement = np.array([[np.float32(cx_raw)], [np.float32(cy_raw)]])
        estimated = kalman.correct(measurement)

        cx = int(estimated[0])
        cy = int(estimated[1])

        # draw centers
        cv2.circle(display, (cx_raw, cy_raw), 4, (0,0,255), -1)   # raw
        cv2.circle(display, (cx, cy), 8, (0,255,0), -1)           # kalman

        # ===============================
        # CENTER LINE (fitLine)
        # ===============================
        if len(c) > 5:
            vx, vy, _, _ = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
            vx = float(vx)
            vy = float(vy)

            length = 1000
            x1 = int(cx - vx * length)
            y1 = int(cy - vy * length)
            x2 = int(cx + vx * length)
            y2 = int(cy + vy * length)

            cv2.line(display, (x1,y1), (x2,y2), (255,0,0), 2)

            angle = np.degrees(np.arctan2(vy, vx))
            cv2.putText(display, f"Angle: {angle:.2f} deg",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # ===============================
    # OBJECT LOST → PREDICT ONLY
    # ===============================
    if not detected and kalman_initialized:
        pred = kalman.predict()
        cx = int(pred[0])
        cy = int(pred[1])
        cv2.circle(display, (cx, cy), 8, (255,0,255), -1)

    # ===============================
    # SHOW
    # ===============================
    cv2.imshow("Detected Object", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()