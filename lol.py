'''
    Trigger-Based ArUco Follower
    © 2025 Arnav Yadavilli. All rights reserved.
'''
import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import math
import sys

# ==========================================
# 1. GPIO & MOTOR SETUP
# ==========================================

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Pin Definitions
LEFT_REAR_A, LEFT_REAR_B = 22, 23
LEFT_FRONT_A, LEFT_FRONT_B = 17, 18
RIGHT_REAR_A, RIGHT_REAR_B = 12, 16
RIGHT_FRONT_A, RIGHT_FRONT_B = 24, 25

PINS = [
    LEFT_REAR_A, LEFT_REAR_B, LEFT_FRONT_A, LEFT_FRONT_B,
    RIGHT_REAR_A, RIGHT_REAR_B, RIGHT_FRONT_A, RIGHT_FRONT_B
]

for pin in PINS:
    GPIO.setup(pin, GPIO.OUT)

# --- TIME-BASED MOTOR FUNCTIONS ---

def forward(duration):
    """Moves forward for 'duration' seconds, then stops."""
    # Left Forward
    GPIO.output(LEFT_REAR_A, GPIO.LOW); GPIO.output(LEFT_REAR_B, GPIO.HIGH)
    GPIO.output(LEFT_FRONT_A, GPIO.LOW); GPIO.output(LEFT_FRONT_B, GPIO.HIGH)
    # Right Forward
    GPIO.output(RIGHT_REAR_A, GPIO.HIGH); GPIO.output(RIGHT_REAR_B, GPIO.LOW)
    GPIO.output(RIGHT_FRONT_A, GPIO.HIGH); GPIO.output(RIGHT_FRONT_B, GPIO.LOW)
    
    time.sleep(duration)
    stop()

def backward(duration):
    """Moves backward for 'duration' seconds, then stops."""
    # Left Backward
    GPIO.output(LEFT_REAR_A, GPIO.HIGH); GPIO.output(LEFT_REAR_B, GPIO.LOW)
    GPIO.output(LEFT_FRONT_A, GPIO.HIGH); GPIO.output(LEFT_FRONT_B, GPIO.LOW)
    # Right Backward
    GPIO.output(RIGHT_REAR_A, GPIO.LOW); GPIO.output(RIGHT_REAR_B, GPIO.HIGH)
    GPIO.output(RIGHT_FRONT_A, GPIO.LOW); GPIO.output(RIGHT_FRONT_B, GPIO.HIGH)

    time.sleep(duration)
    stop()

def right(duration):
    """Spins Right for 'duration' seconds, then stops."""
    # Left Forward / Right Backward
    GPIO.output(LEFT_REAR_A, GPIO.LOW); GPIO.output(LEFT_REAR_B, GPIO.HIGH)
    GPIO.output(LEFT_FRONT_A, GPIO.LOW); GPIO.output(LEFT_FRONT_B, GPIO.HIGH)
    GPIO.output(RIGHT_REAR_A, GPIO.LOW); GPIO.output(RIGHT_REAR_B, GPIO.HIGH)
    GPIO.output(RIGHT_FRONT_A, GPIO.LOW); GPIO.output(RIGHT_FRONT_B, GPIO.HIGH)

    time.sleep(duration)
    stop()

def left(duration):
    """Spins Left for 'duration' seconds, then stops."""
    # Left Backward / Right Forward
    GPIO.output(LEFT_REAR_A, GPIO.HIGH); GPIO.output(LEFT_REAR_B, GPIO.LOW)
    GPIO.output(LEFT_FRONT_A, GPIO.HIGH); GPIO.output(LEFT_FRONT_B, GPIO.LOW)
    GPIO.output(RIGHT_REAR_A, GPIO.HIGH); GPIO.output(RIGHT_REAR_B, GPIO.LOW)
    GPIO.output(RIGHT_FRONT_A, GPIO.HIGH); GPIO.output(RIGHT_FRONT_B, GPIO.LOW)

    time.sleep(duration)
    stop()

def stop():
    """Stops all motors immediately."""
    for pin in PINS:
        GPIO.output(pin, GPIO.LOW)

# ==========================================
# 2. MATH HELPERS
# ==========================================

def get_euler_angles(rvec):
    R, _ = cv2.Rodrigues(rvec)
    pitch = math.atan2(R[2, 1], R[2, 2])
    yaw = math.asin(-R[2, 0])
    roll = math.atan2(R[1, 0], R[0, 0])
    return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)

# ==========================================
# 3. MAIN LOOP
# ==========================================

def main():
    camera_matrix = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=float)
    dist_coeffs = np.zeros((4, 1)) 
    MARKER_SIZE = 0.05 

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    parameters = cv2.aruco.DetectorParameters_create()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # --- TUNING PARAMETERS ---
    TARGET_DIST = 0.50   # Meters
    DIST_BUFFER = 0.05   
    ANGLE_BUFFER = 8.0   
    
    STEP_TURN = 0.1      
    STEP_MOVE = 0.2      

    print("--- SYSTEM BOOT COMPLETE ---")
    print("STANDBY: Waiting for visual confirmation (Show Marker)...")

    # ==========================================
    # PHASE 1: WAIT FOR START SIGNAL
    # ==========================================
    try:
        while True:
            ret, frame = cap.read()
            if not ret: continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            if ids is not None:
                print("\n[CONFIRMED] Marker Detected.")
                print("Engaging Main Drive in 2 seconds...")
                time.sleep(2) # Give you time to put the marker in position
                break # Exit the wait loop and start the main loop
            
            # Reduce CPU usage while waiting
            time.sleep(0.1)

        # ==========================================
        # PHASE 2: MAIN TRACKING LOOP
        # ==========================================
        print("--- MISSION STARTED ---")
        
        while True:
            # Flush buffer
            for _ in range(2): cap.grab()
            
            ret, frame = cap.read()
            if not ret: continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            if ids is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, camera_matrix, dist_coeffs)
                rvec = rvecs[0]
                tvec = tvecs[0]
                
                pitch, yaw, roll = get_euler_angles(rvec)
                dist = tvec[0][2]
                
                sys.stdout.write(f"\rYaw: {yaw:.1f}° | Dist: {dist:.2f}m | Action: ")

                # LOGIC
                if yaw > ANGLE_BUFFER:
                    sys.stdout.write("CORRECTING RIGHT  ")
                    right(STEP_TURN)
                
                elif yaw < -ANGLE_BUFFER:
                    sys.stdout.write("CORRECTING LEFT   ")
                    left(STEP_TURN)
                
                else:
                    if dist > (TARGET_DIST + DIST_BUFFER):
                        sys.stdout.write("APPROACHING       ")
                        forward(STEP_MOVE)
                    elif dist < (TARGET_DIST - DIST_BUFFER):
                        sys.stdout.write("BACKING UP        ")
                        backward(STEP_MOVE)
                    else:
                        sys.stdout.write("HOLDING POSITION  ")
                        stop()
                sys.stdout.flush()

            else:
                stop()
                sys.stdout.write("\r[LOST SIGNAL] Searching...               ")
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop()
        cap.release()
        GPIO.cleanup()

if __name__ == "__main__":
    main()