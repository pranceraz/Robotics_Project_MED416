import cv2
import numpy as np


def main():
    dictionary  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)# check later

    #parameters = cv2.aruco.DetectorParameters()
    MARKER_SIZE = 0.05
    #detector = cv2.aruco.ArucoDetector(dictionary= dictionary,detectorParams= parameters)
    parameters = cv2.aruco.DetectorParameters_create()
    detector = cv2.aruco
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    ret, frame = cap.read()
    h, w = frame.shape[:2]

    # Approximate focal length (usually w or h in pixels) and center
    # focal_length = w 
    # center = (w / 2, h / 2)
    # camera_matrix = np.array(
    #     [[focal_length, 0, center[0]],
    #     [0, focal_length, center[1]],
    #     [0, 0, 1]], dtype="double"
    # )

    camera_matrix = np.array([[600,    0, 320],
                          [0,    600, 240],
                          [0,      0,   1]], dtype=float)

    dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion for this test

    obj_points = np.array([
        [-MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
        [MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
        [MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
        [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0]
    ], dtype=np.float32)


   # dist_coeffs = np.zeros((5, 1))   # or your real distortion coefficients
    while True:
        ret,frame  = cap.read()
        if not ret:
            print("WHYYYYYYYYYYY")
            break
        print(f"{frame.shape} is the shape")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners,ids,rejected = detector.detectMarkers(frame,dictionary,parameters = parameters)

        if ids is not None:
            # Draw the outline of the markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            #for i in range(len(ids)):
                #current_corners = corners[i][0]
            rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE , camera_matrix, dist_coeffs)

            for i in range(len(ids)):
                rvec = rvecs[i]
                tvec = tvecs[i]
                
                try:
                    # 1. Turn the rotation vector (rvec) into a 3x3 matrix
                    R, _ = cv2.Rodrigues(rvec)

                    # 2. Invert the matrix (transpose it)
                    R_inv = np.transpose(R)  # or np.linalg.inv(R)

                    # 3. Calculate Camera Position: -R_inv * tvec
                    # This "undoes" the rotation and translation to find where the camera is
                    camera_position = -np.dot(R_inv, tvec[0])

                    print(f"Camera X: {camera_position[0]:.2f}")
                    print(f"Camera Y: {camera_position[1]:.2f}")
                    print(f"Camera Z: {camera_position[2]:.2f}")
                except Exception as e:
                    print(f"we fked up {e}")
                try:
                    cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
                except AttributeError:
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
                # if success:
                #     # draw 
                #     cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_SIZE)
                #     # Print distance to console (tvec[2] is the Z distance)
                #     print(f"Distance to marker {ids[i][0]}: {tvec[2][0]:.2f} meters")
                z_dist = tvec[0][2]
                cv2.putText(frame, f"Dist: {z_dist:.2f}m", 
                            (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                try:
                    # Pitch (rotation around X)
                    pitch_rad = np.arcsin(-R[2, 0])
                    
                    # Yaw (rotation around Y, most important for steering)
                    yaw_rad = np.arctan2(R[1, 0], R[0, 0])
                    
                    # Roll (rotation around Z)
                    roll_rad = np.arctan2(R[2, 1], R[2, 2])
                    
                    # Convert to Degrees
                    yaw_degrees = np.degrees(yaw_rad)
                    pitch_degrees = np.degrees(pitch_rad)
                    roll_degrees = np.degrees(roll_rad)
                    
                    print(f"Yaw Angle (Horizontal Steering): {yaw_degrees:.2f} degrees")
                    print(f"Pitch Angle (Vertical Tilt): {pitch_degrees:.2f} degrees")
                    print(f"Roll Angle (Camera Twist): {roll_degrees:.2f} degrees")
                    
                except ValueError:
                    print("Error in Euler angle calculation (likely due to gimbal lock at 90 degrees).")
                
        cv2.imshow("ArUco Pose Estimation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()      



if __name__ == "__main__":
    main()