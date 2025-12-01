import cv2
import time
import os

# --- CONFIGURATION ---
NUM_CAPTURES = 20
OUTPUT_DIR = "calibration_images"
# MUST match your actual board: Inner corners 9 points wide, 7 points tall
CHESSBOARD_SIZE = (9, 7) 
# ---------------------

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
print(f"Calibration images will be saved in the '{OUTPUT_DIR}' folder.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Set resolution (recommended for better detection)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

image_count = 0
message = "Press SPACE to Scan"

while image_count < NUM_CAPTURES:
    ret, frame = cap.read()
    if not ret:
        break

    # Display status messages
    cv2.putText(frame, f"Captured: {image_count}/{NUM_CAPTURES}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, message, (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Capture for Calibration", frame)

    key = cv2.waitKey(1) & 0xFF
    
    # Check for capture command
    if key == ord(' '):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- CORE LOGIC: Find Corners ---
        corners_found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        if corners_found:
            # 1. FOUND: Draw corners and save the image
            message = "SUCCESS! Image saved."
            cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners, corners_found)

            filename = os.path.join(OUTPUT_DIR, f"calib_img_{image_count:02d}.jpg")
            cv2.imwrite(filename, frame)
            
            print(f"Captured (Corners Found): {filename}")
            image_count += 1
            
            # Show the frame with drawn corners briefly
            cv2.imshow("Capture for Calibration", frame)
            cv2.waitKey(500) 
            
        else:
            # 2. NOT FOUND: Inform the user
            message = "FAILED! Move board and try again."
            print("Failed to find corners. Try a different angle/distance.")
        
        time.sleep(0.2) # Short pause before next scan

    # Exit loop when 'q' is pressed
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nFinished capture. {image_count} images saved.")