import cv2
import numpy as np
import driver
import os
import time

CALIB_FILE = 'camera_params.npz'
KP_YAW = 2.0          
YAW_TOLERANCE = 3.0
TURN_ANGLE = 30.0
SEGMENT_DISTANCE = 0.24
MARKER_SIZE = 0.05
def load_calibration_parameters():
    """Loads camera matrix and distortion coefficients from the .npz file."""
    import os
    CALIB_FILE = "camera_params.npz"

    if not os.path.exists(CALIB_FILE):
        print(f"ERROR: Calibration file '{CALIB_FILE}' not found.")
        print("Please run the calibration script first!")
        # Use dummy values as a fallback
        return None, None

    data = np.load(CALIB_FILE)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    return camera_matrix, dist_coeffs

def get_pose_data(frame, dictionary, parameters, camera_matrix, dist_coeffs, marker_size):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    
    if ids is not None and len(ids) == 1:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
        rvec, tvec = rvecs[0], tvecs[0]
        R, _ = cv2.Rodrigues(rvec)
        
        yaw_rad = np.arctan2(R[1, 0], R[0, 0])
        yaw_degrees = np.degrees(yaw_rad)
        z_dist = tvec[0][2]
        
        return z_dist, yaw_degrees, frame, rvec, tvec, R
    
    return None, None, frame, None, None, None


def turn_to_target_yaw(target_yaw, current_yaw):
    """Controls the robot to align with the target yaw angle."""
    yaw_error = target_yaw - current_yaw
    turn_command = yaw_error * KP_YAW
    
    if abs(yaw_error) > YAW_TOLERANCE:
        if turn_command > 0:
            # Target is to the right
            driver.right()
        else:
            # Target is to the left
            driver.left()
        return False # Still turning
    else:
        driver.stop()
        return True # Alignment achieved

def main():
    # Load calibration data (REPLACES HARDCODED VALUES)
    camera_matrix, dist_coeffs = load_calibration_parameters()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    parameters = cv2.aruco.DetectorParameters_create()

    if camera_matrix is None:
        # If loading failed, exit the program
        return

    # Initialize camera
    cap = cv2.VideoCapture(0)
    time.sleep(1)
    if not cap.isOpened():
        print("Failed to open camera.")
        return

    # --- PATH FOLLOWING SETUP ---
    turn_sequence = [TURN_ANGLE, -TURN_ANGLE, TURN_ANGLE] # Right, Left, Right
    current_segment = 0
    segments_to_travel = len(turn_sequence) + 1 # 4 segments total
    state = "INITIAL_SETUP"
    
    target_z_dist = None
    
    try:
        while current_segment < segments_to_travel:
            ret, frame = cap.read()
            if not ret:
                print("Lost camera feed.")
                break

            z_dist, yaw_degrees, frame, rvec, tvec, R = get_pose_data(
                frame, dictionary, parameters, camera_matrix, dist_coeffs, MARKER_SIZE
            )
            
            while z_dist is None:
                print("Marker lost. Waiting for it to appear...")
                driver.stop()  # ensure the robot stays still
                ret, frame = cap.read()
                if not ret:
                    print("Lost camera feed while waiting for marker.")
                    break
                
                z_dist, yaw_degrees, frame, rvec, tvec, R = get_pose_data(
                    frame, dictionary, parameters, camera_matrix, dist_coeffs, MARKER_SIZE
                )
                
                # Optional: show camera feed for debugging
                cv2.imshow("Waiting for Marker", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                
            # INITIAL_SETUP: Determine the starting point and transition to ALIGNMENT
            if state == "INITIAL_SETUP":
                z_dist_start = z_dist 
                target_z_dist = z_dist_start - SEGMENT_DISTANCE
                print(f"Initial Z: {z_dist_start:.2f}m. Target Z for first segment: {target_z_dist:.2f}m.")
                state = "ALIGNMENT"

            # ----------------------------------------------------
            # 1. ALIGNMENT STATE (Turning to face marker)
            # ----------------------------------------------------
            if state == "ALIGNMENT":
                print(f"STATE: ALIGNMENT (Yaw: {yaw_degrees:.1f} deg)")
                
                is_aligned = turn_to_target_yaw(0.0, yaw_degrees) 
                
                if is_aligned:
                    print(f"ALIGNMENT ACHIEVED for segment {current_segment + 1}.")
                    state = "NAVIGATION"
            
            # ----------------------------------------------------
            # 2. NAVIGATION STATE (Moving Forward)
            # ----------------------------------------------------
            elif state == "NAVIGATION":
                print(f"STATE: NAVIGATION (Z: {z_dist:.2f}m, Target Z: {target_z_dist:.2f}m)")
                
                if z_dist > target_z_dist:
                    # Move forward while adjusting yaw slightly
                    driver.forward()
                else:
                    # Segment distance reached! Stop and prepare for the next turn.
                    driver.stop()
                    current_segment += 1
                    print(f"SEGMENT {current_segment} complete.")
                    
                    if current_segment < len(turn_sequence):
                        # Execute the physical turn (e.g., 30 deg Right or Left)
                        turn_angle_to_make = turn_sequence[current_segment - 1] # Use the executed turn angle
                        
                        if turn_angle_to_make > 0:
                            print(f"Executing {TURN_ANGLE} degrees RIGHT.")
                            driver.right(sleep_time=1.0) 
                        else:
                            print(f"Executing {TURN_ANGLE} degrees LEFT.")
                            driver.left(sleep_time=1.0) 

                        # Update the target distance for the next segment
                        target_z_dist = z_dist - SEGMENT_DISTANCE
                        
                        state = "ALIGNMENT" # Go back to alignment to correct facing direction
            
            # --- Visualization and Debug ---
            # Draw axis and distance text
            try:
                cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
            except AttributeError:
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

            cv2.putText(frame, f"Dist: {z_dist:.2f}m", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Yaw: {yaw_degrees:.1f}deg", (150, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            cv2.imshow("ArUco Navigation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print("\n--- PATH COMPLETE ---")

    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
        driver.stop()
        driver.GPIO.cleanup()

# def main():
#     dictionary  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)# check later

#     #parameters = cv2.aruco.DetectorParameters()
#     MARKER_SIZE = 0.05
#     #detector = cv2.aruco.ArucoDetector(dictionary= dictionary,detectorParams= parameters)
#     parameters = cv2.aruco.DetectorParameters_create()
#     detector = cv2.aruco
#     cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
#     ret, frame = cap.read()
#     h, w = frame.shape[:2]

    
#     # Approximate focal length (usually w or h in pixels) and center
#     # focal_length = w 
#     # center = (w / 2, h / 2)
#     # camera_matrix = np.array(
#     #     [[focal_length, 0, center[0]],
#     #     [0, focal_length, center[1]],
#     #     [0, 0, 1]], dtype="double"
#     # )

#     camera_matrix = np.array([[600,    0, 320],
#                           [0,    600, 240],
#                           [0,      0,   1]], dtype=float)

#     dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion for this test

#     obj_points = np.array([
#         [-MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
#         [MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
#         [MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
#         [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0]
#     ], dtype=np.float32)


#    # dist_coeffs = np.zeros((5, 1))   # or your real distortion coefficients
#     while True:
#         ret,frame  = cap.read()
#         if not ret:
#             print("WHYYYYYYYYYYY")
#             break
#         print(f"{frame.shape} is the shape")
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         corners,ids,rejected = detector.detectMarkers(frame,dictionary,parameters = parameters)

#         if ids is not None:
#             # Draw the outline of the markers
#             cv2.aruco.drawDetectedMarkers(frame, corners, ids)
#             #for i in range(len(ids)):
#                 #current_corners = corners[i][0]
#             rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE , camera_matrix, dist_coeffs)

#             for i in range(len(ids)):
#                 rvec = rvecs[i]
#                 tvec = tvecs[i]
                
#                 try:
#                     # 1. Turn the rotation vector (rvec) into a 3x3 matrix
#                     R, _ = cv2.Rodrigues(rvec)

#                     # 2. Invert the matrix (transpose it)
#                     R_inv = np.transpose(R)  # or np.linalg.inv(R)

#                     # 3. Calculate Camera Position: -R_inv * tvec
#                     # This "undoes" the rotation and translation to find where the camera is
#                     camera_position = -np.dot(R_inv, tvec[0])

#                     print(f"Camera X: {camera_position[0]:.2f}")
#                     print(f"Camera Y: {camera_position[1]:.2f}")
#                     print(f"Camera Z: {camera_position[2]:.2f}")
#                 except Exception as e:
#                     print(f"we fked up {e}")
#                 try:
#                     cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
#                 except AttributeError:
#                     cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
#                 # if success:
#                 #     # draw 
#                 #     cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_SIZE)
#                 #     # Print distance to console (tvec[2] is the Z distance)
#                 #     print(f"Distance to marker {ids[i][0]}: {tvec[2][0]:.2f} meters")
#                 z_dist = tvec[0][2]
#                 cv2.putText(frame, f"Dist: {z_dist:.2f}m", 
#                             (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                 try:
#                     # Pitch (rotation around X)
#                     pitch_rad = np.arcsin(-R[2, 0])
                    
#                     # Yaw (rotation around Y, most important for steering)
#                     yaw_rad = np.arctan2(R[1, 0], R[0, 0])
                    
#                     # Roll (rotation around Z)
#                     roll_rad = np.arctan2(R[2, 1], R[2, 2])
                    
#                     # Convert to Degrees
#                     yaw_degrees = np.degrees(yaw_rad)
#                     pitch_degrees = np.degrees(pitch_rad)
#                     roll_degrees = np.degrees(roll_rad)
                    
#                     print(f"Yaw Angle (Horizontal Steering): {yaw_degrees:.2f} degrees")
#                     print(f"Pitch Angle (Vertical Tilt): {pitch_degrees:.2f} degrees")
#                     print(f"Roll Angle (Camera Twist): {roll_degrees:.2f} degrees")
                    
#                 except ValueError:
#                     print("Error in Euler angle calculation (likely due to gimbal lock at 90 degrees).")
                
#         cv2.imshow("ArUco Pose Estimation", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()      



if __name__ == "__main__":
    main()import cv2