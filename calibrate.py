import cv2
import numpy as np
import glob

# =================================
# CONFIG (ปรับตาม chessboard จริง)
# =================================
chessboard_size = (9, 6)   # inner corners (จำนวนจุดตัดด้านใน)
square_size = 0.025        # เมตร (เช่น 2.5 cm = 0.025)

# =================================
# เตรียม world coordinate
# =================================
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []

# =================================
# โหลดภาพ calibration
# =================================
images = glob.glob("calib_images/*.jpg")

print("found images:", len(images))

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow("corners", img)
        cv2.waitKey(200)

cv2.destroyAllWindows()

# =================================
# คำนวณ calibration
# =================================
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

print("\nCamera matrix:\n", camera_matrix)
print("\nDistortion:\n", dist_coeffs)

# =================================
# SAVE (ตรงกับ YOLO script คุณ)
# =================================
np.save("camera_matrix.npy", camera_matrix)
np.save("dist_coeffs.npy", dist_coeffs)

print("\nSaved camera_matrix.npy and dist_coeffs.npy")