import numpy as np
import cv2
import glob

# --- CONFIGURATION ---
# IMPORTANT: These are the number of *internal corners* (e.g., 8x8 board has 7x7 internal corners)
CHECKERBOARD = (7, 7) 
# Size of one square in your chosen unit (e.g., meters or millimeters).
# This sets the scale for your pose estimation.
SQUARE_SIZE = 0.025 # Example: 25 mm or 2.5 cm

# Termination criteria for the corner refinement process
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Prepare "object points" (3D coordinates of the corners relative to the board)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
# Create a grid of (x, y, 0) coordinates based on the square size
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

# Path to your calibration images (e.g., 10-20 images of the chessboard from different angles)
images = glob.glob('calibration_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # If corners are found, add object points and refine image points
    if ret:
        objpoints.append(objp)
        
        # Refine corner pixel coordinates
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners to verify detection (optional)
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500) # Wait 0.5 seconds between images

cv2.destroyAllWindows()

# The image size (width, height) is required for calibration
height, width = gray.shape[:2]

# Perform the actual calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,        # 3D points (real world)
    imgpoints,        # 2D points (pixels)
    (width, height),  # Image size
    None,             # Initial camera matrix (optional)
    None              # Initial distortion coefficients (optional)
)

# --- RESULTS ---
print("Calibration Successful:", ret)
print("\n--- Camera Matrix (mtx) ---")
print(mtx)
print("\n--- Distortion Coefficients (dist) ---")
print(dist)

# --- SAVE THE RESULTS ---
# You must save these two arrays (mtx and dist) to a file (like NPZ)
# so you can load them later for ArUco pose estimation.
np.savez("camera_calibration_data.npz", mtx=mtx, dist=dist)
print("\nCalibration data saved to camera_calibration_data.npz")