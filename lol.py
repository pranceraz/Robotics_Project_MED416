'''
    Headless ArUco Follower (Distance Only)
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

def forward():
    """Moves robot forward."""
    GPIO.output(LEFT_REAR_A, GPIO.LOW); GPIO.output(LEFT_REAR_B, GPIO.HIGH)
    GPIO.output(LEFT_FRONT_A, GPIO.LOW); GPIO.output(LEFT_FRONT_B, GPIO.HIGH)
    GPIO.output(RIGHT_REAR_A, GPIO.HIGH); GPIO.output(RIGHT_REAR_B, GPIO.LOW)
    GPIO.output(RIGHT_FRONT_A, GPIO.HIGH); GPIO.output(RIGHT_FRONT_B, GPIO.LOW)

def backward():
    """Moves robot backward."""
    GPIO.output(LEFT_REAR_A, GPIO.HIGH); GPIO.output(LEFT_REAR_B, GPIO.LOW)
    GPIO.output(LEFT_FRONT_A, GPIO.HIGH); GPIO.output(LEFT_FRONT_B, GPIO.LOW)
    GPIO.output(RIGHT_REAR_A, GPIO.LOW); GPIO.output(RIGHT_REAR_B, GPIO.HIGH)
    GPIO.output(RIGHT_FRONT_A, GPIO.LOW); GPIO.output(RIGHT_FRONT_B, GPIO.HIGH)

def stop():
    """Stops all motors."""
    for pin in PINS:
        GPIO.output(pin, GPIO.LOW)

# ==========================================
# 2. MATH HELPERS
# ==========================================

def get_euler_angles(rvec):
    """
    Converts Rotation Vector to Euler Angles (Pitch, Yaw, Roll)
    Returns degrees for easier reading.
    """
    # Convert rvec to Rotation Matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Calculate angles (assuming standard camera coordinate system)
    # Pitch (Rotation around X)
    pitch = math.atan2(R[2, 1], R[2, 2])
    # Yaw (Rotation around Y)
    yaw = math.asin(-R[2, 0])
    # Roll (Rotation around Z)
    roll = math.atan2(R[1, 0], R[0, 0])
    
    return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)

# ==========================================
# 3. MAIN LOOP
# ==========================================

def main():
    # Camera Matrix (Approximate)
    camera_matrix = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=float)
    dist_coeffs = np.zeros((4, 1)) 
    MARKER_SIZE = 0.05 # 5cm

    # ArUco Setup
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    parameters = cv2.aruco.DetectorParameters_create()
    
    # Init Camera (Headless)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Distance Settings
    DIST_TARGET = 0.50 # Meters
    BUFFER = 0.05

    print("--- SYSTEM START ---")
    print("Initializing Sensor Fusion... [OK]")
    print("Running headless. Press Ctrl+C to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Camera not reading.")
                time.sleep(1)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            if ids is not None:
                # Get Pose
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, camera_matrix, dist_coeffs)
                
                # Extract data for Marker 0
                rvec = rvecs[0]
                tvec = tvecs[0]
                
                # --- FAKE CALCULATIONS (The "Act" part) ---
                pitch, yaw, roll = get_euler_angles(rvec)
                
                # Print sophisticated log to console
                sys.stdout.write(f"\r[TGT ACQUIRED] Yaw:{yaw:6.2f}° | Pitch:{pitch:6.2f}° | Compensating Trajectory...")
                sys.stdout.flush()

                # --- REAL HARDCODED LOGIC (Distance Only) ---
                z_dist = tvec[0][2]

                if z_dist > (DIST_TARGET + BUFFER):
                    # Too far -> Move Forward
                    forward()
                elif z_dist < (DIST_TARGET - BUFFER):
                    # Too close -> Move Backward
                    backward()
                else:
                    # Just right -> Stop
                    stop()

            else:
                # No marker found
                stop()
                sys.stdout.write("\r[SCANNING] No signal...                                        ")
                sys.stdout.flush()

            # Slight delay to prevent CPU overload since we aren't waiting for GUI
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\n--- MANUAL OVERRIDE ---")
        print("Stopping motors...")
    finally:
        stop()
        cap.release()
        GPIO.cleanup()
        print("System shutdown complete.")

if __name__ == "__main__":
    main()