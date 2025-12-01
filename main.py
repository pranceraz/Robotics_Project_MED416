import cv2
import numpy as np
import driver


KP_YAW = 2.0          
YAW_TOLERANCE = 3.0
TURN_ANGLE = 30.0
SEGMENT_DISTANCE = 0.24

def get_pose_data(frame, dictionary, parameters, camera_matrix, dist_coeffs, marker_size):
    """Detects marker, estimates pose, and extracts Yaw, Z-distance."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    
    if ids is not None and len(ids) == 1:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
        
        rvec, tvec = rvecs[0], tvecs[0]
        R, _ = cv2.Rodrigues(rvec)
        
        # Yaw Angle calculation (Rotation around Y-axis)
        try:
            # Yaw (rotation around Y, most important for steering)
            yaw_rad = np.arctan2(R[1, 0], R[0, 0])
            yaw_degrees = np.degrees(yaw_rad)
        except ValueError:
            yaw_degrees = 0 # Gimbal lock or error
        
        # Distance (Z-axis)
        z_dist = tvec[0][2]
        
        return z_dist, yaw_degrees, frame
        
    return None, None, frame

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

 Pose Detection with an aruco marker

Analyzing ArUco Pose

I'm now diving into the core principles of pose detection, with a particular focus on ArUco markers. I need to understand coordinate systems and how camera calibration fits in. Currently I'm looking into OpenCV implementations.

Defining Detection Steps

Now I'm focusing on the functions critical for ArUco pose estimation, namely cv2.aruco.detectMarkers and cv2.aruco.estimatePoseSingleMarkers. I'm exploring the necessary inputs, such as the camera matrix and distortion coefficients, and how to get these parameters. Ultimately, I will produce a detailed guide on pose detection using an ArUco marker, complete with code and mathematical explanations.

Defining Core Concepts

I've established a solid framework for addressing the user's need. I'm comfortable with the underlying concepts, including ArUco markers, dictionaries, and pose estimation. My focus is now on applying these concepts to the specific problem at hand.

Refining Implementation Strategy

I'm now diving into the technical execution. My research on OpenCV's functions solidified my plan. While estimatePoseSingleMarkers is familiar, it's potentially outdated. I'll focus on solvePnP for accurate, current implementation. Ensuring the user receives the most relevant and updated information regarding OpenCV versions is my top priority.

Updating Coordinate Systems

My thoughts now revolve around providing a clear and simple explanation of the coordinate system. I'm focusing on clarifying the Z-axis orientation relative to the camera and ensuring the user can easily understand it. I'm also considering the potential difference between the newer and older OpenCV versions (4.x vs pre-4.x) and how they could affect the implementation. I am investigating the new ArucoDetector class.

Validating Coordinate Conventions

I'm verifying the coordinate system convention, specifically the Z-axis direction relative to the camera, to ensure clarity and avoid ambiguity. I've found initial references to the RGB axis convention, but I'm confirming this is the standard. I'm also double-checking recent OpenCV ArUco API changes, focusing on the ArucoDetector class, to provide users with the most up-to-date and relevant information for implementation. My priority is to keep the information relevant and correct.

Evaluating API Updates

I've completed a thorough investigation of the OpenCV ArUco module. I'm focusing on the deprecation of estimatePoseSingleMarkers. It seems newer versions lean towards different methods, and I must determine the best course of action to maintain compatibility while leveraging the latest features.

Modernizing the Solution

I'm now zeroing in on integrating cv2.solvePnP as the recommended approach, emphasizing its robustness. Though detectMarkers remains widely applicable, I'll spotlight the ArucoDetector for recent OpenCV users. This involves explaining the necessity of camera calibration to unlock 3D data from 2D pixel coordinates and to de-warp images. I'll stick with detectMarkers for core examples, but provide guidance on how to upgrade for a future-proofed approach, and will highlight objdetect's potential role as an advanced note.

Integrating Core Principles

I'm now deeply immersed in the coordinate systems, ensuring their clarity. I've defined the camera and marker frame coordinate systems, clarifying their relationship, as well as the units used in solvePnP. This step builds a comprehensive understanding for all users, regardless of how they access the function. I'm prioritizing accuracy and intuitive application.

Developing Comprehensive Examples

I'm now focusing on integrating cv2.drawFrameAxes into example code that uses solvePnP. This step will demonstrate how to visualize the pose estimation results. My goal is to deliver an in-depth, instructive explanation, that's more than just providing a code snippet. The emphasis will be on conveying a conceptual understanding of the process.

Pose detection with an ArUco marker is a method to find the 3D position (translation) and orientation (rotation) of a marker relative to your camera. In computer vision, this is often called "Pose Estimation."

For this to work mathematically, your code needs to know two things:

    The physical size of the marker (e.g., "this square is exactly 5 cm wide").

    The camera's internal properties (Intrinsic Matrix and Distortion Coefficients).

The Modern Approach (OpenCV 4.7+)

While older tutorials use estimatePoseSingleMarkers, that function is now deprecated. The standard, robust way to do this in modern OpenCV is using cv2.solvePnP. This function calculates the pose that best maps the 3D points of the marker to the 2D pixels detected in the image.

Prerequisites

You must have a calibrated camera. If you do not have your camera's cameraMatrix (3x3 matrix) and distCoeffs (distortion vector), the distance and rotation calculations will be mathematically wrong.

    Tip: If you just want to test the code without calibrating, you can approximate the camera matrix (shown in the example below), but your depth (Z-axis) measurements will be inaccurate.

Python Implementation

Here is a complete script that detects a marker and estimates its pose using solvePnP.
Python

import cv2
import numpy as np

# --- 1. CONFIGURATION ---
# Define the marker size in meters (or any unit you want the output to be in)
MARKER_SIZE = 0.05  # 5cm
# Pick the dictionary used to generate the marker
marker_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
param_markers = cv2.aruco.DetectorParameters()

# --- 2. CAMERA CALIBRATION (REPLACE WITH YOUR ACTUAL DATA) ---
# You should load these from a file after calibrating your camera.
# This is a "dummy" calibration for testing if you don't have one.
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
h, w = frame.shape[:2]

# Approximate focal length (usually w or h in pixels) and center
focal_length = w 
center = (w / 2, h / 2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]],
     [0, focal_length, center[1]],
     [0, 0, 1]], dtype="double"
)
dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion for this test

# --- 3. DEFINE THE 3D MARKER COORDINATES ---
# We define the 4 corners of the marker in its own 3D coordinate system.
# We assume the marker is flat on the Z=0 plane.
# Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
obj_points = np.array([
    [-MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
    [MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
    [MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
    [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0]
], dtype=np.float32)

print("Starting video stream. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejected = cv2.aruco.detectMarkers(
        gray_frame, marker_dict, parameters=param_markers
    )

    if ids is not None:
        # Draw the outline of the markers
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Loop through each detected marker
        for i in range(len(ids)):
            # The corners of the current marker
            current_corners = corners[i][0]
            
            # --- SOLVE PNP ---
            # Match the 3D points (obj_points) to the 2D image points (current_corners)
            success, rvec, tvec = cv2.solvePnP(
                obj_points, 
                current_corners, 
                camera_matrix, 
                dist_coeffs, 
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            
            if success:
                # Draw the axes (X: Red, Y: Green, Z: Blue)
                # length of the axis lines in the same unit as MARKER_SIZE
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_SIZE)
                
                # Print distance to console (tvec[2] is the Z distance)
                # print(f"Distance to marker {ids[i][0]}: {tvec[2][0]:.2f} meters")

    cv2.imshow("ArUco Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

Understanding the Output

When solvePnP runs, it outputs two vectors:

    tvec (Translation Vector): A 3x1 vector [x, y, z].

        X: Horizontal position relative to the camera center.

        Y: Vertical position relative to the camera center.

        Z: Forward distance from the camera. This is usually the most important value for distance measurement.

    rvec (Rotation Vector): A 3x1 vector representing the axis of rotation and the angle. To make this human-readable (like Euler angles: yaw, pitch, roll), you can convert it using cv2.Rodrigues(rvec).

Common Pitfalls

    Wrong Dictionary: If you generated a "4x4_50" marker but your code looks for "6x6_250", nothing will be detected.

    Coordinate System: In OpenCV, the Z-axis points forward from the camera. The X-axis points right, and the Y-axis points down.

    Marker Size Units: If you define MARKER_SIZE = 5 (meaning 5cm) your tvec output will be in centimeters. If you define it as 0.05 (meters), your output will be in meters.

Would you like to know how to calculate the Yaw, Pitch, and Roll angles from the rvec specifically?

ArUco Marker Pose Estimation in OpenCV

This video is relevant because it provides a visual walkthrough of the pose estimation process using Python and OpenCV, reinforcing the code concepts above with real-time examples.
ArUco Marker Pose Estimation in OpenCV - YouTube
Nicolai Nielsen · 26K views

im on a raspberry pi 4 so use the older version of open cv

Running ArUco detection on a Raspberry Pi 4 often involves older versions of OpenCV (like 4.5.x or older) where the syntax is different. Specifically, you likely need to use the aruco submodule directly rather than the newer objdetect API.

The "Legacy" Approach (OpenCV < 4.7)

In older versions, you use estimatePoseSingleMarkers, which calculates the pose for you in one line (instead of using solvePnP manually).

Prerequisites for Raspberry Pi

On the Pi, the standard opencv-python package often lacks the ArUco modules. You usually need the "contrib" version:
Bash

pip install opencv-contrib-python

Python Script (Legacy Syntax)

This script uses cv2.aruco.Dictionary_get, cv2.aruco.DetectorParameters_create, and cv2.aruco.estimatePoseSingleMarkers.
Python

import cv2
import numpy as np

# --- 1. CONFIGURATION ---
marker_size = 0.05  # 5cm (Adjust this to your actual marker size)
calib_data_path = "calib_data.npz" # Path to your calibration file

# --- 2. CAMERA SETUP ---
cap = cv2.VideoCapture(0) # Use 0 for USB webcam. For PiCamera, see note below.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- 3. LOAD CALIBRATION ---
# (Using dummy values if no file is found - BUT accurate pose needs real calibration)
try:
    with np.load(calib_data_path) as X:
        camera_matrix, dist_coeffs = [X[i] for i in ('cameraMatrix', 'distCoeffs')]
    print("Calibration data loaded successfully.")
except:
    print("WARNING: Using dummy calibration. Distances will be wrong!")
    w, h = 640, 480
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))

# --- 4. OLDER ARUCO SETUP ---
# In older versions, we use Dictionary_get and DetectorParameters_create
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters_create()

print("Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejected = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters
    )

    if ids is not None:
        # Draw outlines
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # --- LEGACY POSE ESTIMATION ---
        # This function was deprecated in 4.7+ but works in older versions
        # It returns a list of rvecs and tvecs (one for each marker)
        rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_size, camera_matrix, dist_coeffs
        )

        for i in range(len(ids)):
            # Unpack the vectors for this specific marker
            rvec = rvecs[i]
            tvec = tvecs[i]

            # Draw Axes
            # Note: 'drawAxis' is the old function. If it fails, try 'drawFrameAxes'
            try:
                cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
            except AttributeError:
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

            # Print distance (Z-axis)
            # tvec is usually shape (1, 3) or (3, 1), so we access the Z index carefully
            z_dist = tvec[0][2]
            cv2.putText(frame, f"Dist: {z_dist:.2f}m", 
                        (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Pi ArUco', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

Key Differences for Raspberry Pi / Older OpenCV

    estimatePoseSingleMarkers: This function is the "magic bullet" in older versions. It takes the corners and marker size and spits out the rotation (rvec) and translation (tvec) directly.

    drawAxis: Older versions use cv2.aruco.drawAxis. Newer versions renamed this to cv2.drawFrameAxes. The code above handles both.

    FPS on Pi: If you find cv2.VideoCapture(0) is slow (laggy), the Pi often performs better if you use the Picamera2 library or threaded video streams (like imutils.video.VideoStream), but the code above is the simplest starting point.

Would you like me to explain how to optimize the frame rate on the Raspberry Pi 4 by creating a threaded video stream helper?

ArUco Marker Tracking with Raspberry Pi

This video is relevant because it specifically demonstrates ArUco marker tracking on a Raspberry Pi, covering the practical setup and performance expectations on that specific hardware.

z_dist = tvec[0][2]

cv2.putText(frame, f"Dist: {z_dist:.2f}m",

(int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10),

cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

wtf is this also where is the pose of the of the camera

That code block is doing two things: extracting the distance and then drawing it on the screen.

1. WTF is z_dist = tvec[0][2]?

It extracts the distance from the camera to the marker.

    tvec (Translation Vector): This variable holds the position of the marker [x, y, z].

        Shape: In the older OpenCV version you are using, tvec usually comes in a weird shape like (1, 3) (a list containing one list of 3 numbers).

        tvec[0]: Grabs the inner list [x, y, z].

        tvec[0][2]: Grabs the 3rd number (Z).

    Why Z? In OpenCV's camera system, Z is the axis pointing straight out of the lens. Therefore, Z = Distance.

2. WTF is cv2.putText(...)?

This draws the text string "Dist: 1.54m" onto the video frame.

    frame: The image to draw on.

    f"Dist: {z_dist:.2f}m": The text to write (e.g., "Dist: 0.50m").

    (int(corners[...]), int(corners[...]) - 10): The Coordinate.

        It grabs the top-left corner of the marker (x, y) and subtracts 10 pixels from y.

        This ensures the text sits just above the marker, rather than covering it.

    cv2.FONT_HERSHEY_SIMPLEX: The font type.

    0.5: The font size (scale).

    (0, 255, 0): The color (Green in BGR).

    2: The thickness of the letters.

"Where is the pose of the camera?"

This is the most confusing part of computer vision. tvec and rvec do NOT give you the pose of the camera. They give you the pose of the marker.

    What OpenCV gives you: "The marker is 2 meters forward and rotated 30 degrees relative to the camera."

    What you probably want: "The camera is at coordinate (X,Y,Z) in the room."

To get the Camera's Pose (position in the world), you have to mathematically invert the marker pose. You need to rotate the world backwards and translate it backwards.

Here is the exact code to find the Camera's Position:
Python

# Inside your loop, after getting rvec and tvec
import numpy as np

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

What do these numbers mean now?

    Camera X: How far left/right the camera is from the marker's center.

    Camera Y: How high/low the camera is from the marker.

    Camera Z: How far away the camera is (should match the distance you saw before, but now calculated from the marker's perspective).

okay i need to find the from z s projection onto the cameras parallel plane with th edirection that the camera is pointing at in the new x-Z plane

That's a very specific request! You are asking for the components of the marker's Zworld​ direction vector as it is seen in the camera's X and Z axes. This projection shows the yaw (horizontal tilt) of the marker's Z-axis relative to the camera's viewing axis.

Here is how you get that projected vector, which we'll call Pxz​:

1. Transform the World Z-axis to the Camera Frame

The rvec you get from estimatePoseSingleMarkers describes the rotation from the World Frame (Marker) to the Camera Frame. The direction of the Zworld​ axis in the camera frame (d) is given by the third column of the rotation matrix R.

We use cv2.Rodrigues to convert the rotation vector to a matrix.
Python

# Inside your detection loop, after you get rvec:
import numpy as np

# 1. Convert rvec (3x1 vector) to R (3x3 rotation matrix)
R_matrix, _ = cv2.Rodrigues(rvec)

# 2. The direction vector 'd' of the World Z-axis in the Camera Frame is 
#    the third column of R_matrix.
#    d = R_matrix * [0, 0, 1]T
d_x = R_matrix[0, 2] # X component (right/left)
d_y = R_matrix[1, 2] # Y component (up/down)
d_z = R_matrix[2, 2] # Z component (forward/depth)

# The full direction vector is d = [d_x, d_y, d_z]
print(f"World Z-axis direction vector in Camera Frame (d): [{d_x:.3f}, {d_y:.3f}, {d_z:.3f}]")

2. Project onto the New X−Z Plane

The projection of the vector d onto the camera's X−Z plane is simply its X and Z components.

The resulting vector in the new X′−Z′ plane is:
Pxz​=[dx​dz​​]

    dx​ (the X′ component) tells you how much the world Z-axis tilts horizontally (Yaw).

    dz​ (the Z′ component) tells you how much the world Z-axis points into the camera's Z-axis.

Example Code for the Projection:
Python

# Continuing from the previous snippet:

# The vector projected onto the camera's X-Z plane
P_xz = np.array([d_x, d_z])

print(f"Projected Vector P_xz (X', Z'): [{P_xz[0]:.3f}, {P_xz[1]:.3f}]")

# You can also get the angle of this vector from the Camera's Z-axis:
# The angle of the projection onto the X-Z plane (Yaw Angle)
yaw_angle_radians = np.arctan2(d_x, d_z)
yaw_angle_degrees = np.degrees(yaw_angle_radians)

print(f"Yaw Angle (from Camera Z-axis): {yaw_angle_degrees:.2f} degrees")

Camera Frame Component	Physical Meaning
dx​ (d_x)	Right/Left tilt
dy​ (d_y)	Up/Down tilt (used for Pitch)
dz​ (d_z)	Forward/Depth alignment

The combination Pxz​=[dx​,dz​] effectively isolates the horizontal rotation and ignores the vertical pitch component (dy​).

okay no basically I want to do angular movement based on the aruco marker

That is a fantastic application! To implement angular movement based on the marker, you need to extract the Euler angles (Yaw, Pitch, Roll) from the rotation vector (rvec) you already calculated.

The Yaw angle is the most crucial for horizontal angular movement (steering a robot left or right to face the marker).

Here is how you convert the rotation vector into human-readable angles and apply simple control logic.

1. Extracting Yaw, Pitch, and Roll

The rvec is an axis-angle representation. We must convert it to the R matrix first, and then to Euler angles (in degrees) to easily interpret the marker's orientation.

We will use the standard ZYX Euler convention (which defines yaw, pitch, and roll in relation to the camera's axes):
Python

# Assuming rvec is already defined for a detected marker
# rvec, tvec = rvecs[i], tvecs[i] 

# 1. Convert rvec (3x1) to R matrix (3x3)
R_matrix, _ = cv2.Rodrigues(rvec)

# 2. Extract Euler Angles (ZYX Convention)
# Note: These calculations can have singularities (gimbal lock) but work well 
# for most practical marker orientations.
try:
    # Pitch (rotation around X)
    pitch_rad = np.arcsin(-R_matrix[2, 0])
    
    # Yaw (rotation around Y, most important for steering)
    yaw_rad = np.arctan2(R_matrix[1, 0], R_matrix[0, 0])
    
    # Roll (rotation around Z)
    roll_rad = np.arctan2(R_matrix[2, 1], R_matrix[2, 2])
    
    # Convert to Degrees
    yaw_degrees = np.degrees(yaw_rad)
    pitch_degrees = np.degrees(pitch_rad)
    roll_degrees = np.degrees(roll_rad)
    
    print(f"Yaw Angle (Horizontal Steering): {yaw_degrees:.2f} degrees")
    print(f"Pitch Angle (Vertical Tilt): {pitch_degrees:.2f} degrees")
    print(f"Roll Angle (Camera Twist): {roll_degrees:.2f} degrees")
    
except ValueError:
    print("Error in Euler angle calculation (likely due to gimbal lock at 90 degrees).")

2. Implementing Angular Movement Control

For a control system (like a robot motor driver), you typically use the Yaw angle to determine the direction and speed of rotation.

A Simple P-Controller Concept

A Proportional (P) controller is the easiest way to link the angle error to a movement command.

    Set Point: You want the Yaw angle to be 0 degrees (the marker is directly ahead).

    Error: Error=Yaw_degrees−0.

    Control Signal: Turn_Speed=Error×Kp​

Here is the implementation:
Python

# --- Control Parameters ---
KP = 0.5  # Proportional gain (TUNE THIS VALUE!)
MAX_SPEED = 50 # Max motor speed (e.g., 0 to 100)

# 1. Get the current Yaw Angle (from the previous calculation)
current_yaw = yaw_degrees

# 2. Calculate the Error
yaw_error = current_yaw  # Target is 0 degrees

# 3. Calculate the Control Signal (Turn Speed)
# The sign of the error tells us the direction:
# Positive error -> Marker is to the right of center -> Need to turn RIGHT
# Negative error -> Marker is to the left of center -> Need to turn LEFT

turn_command = yaw_error * KP

# 4. Limit the Speed (Clamp the value)
if turn_command > MAX_SPEED:
    turn_command = MAX_SPEED
elif turn_command < -MAX_SPEED:
    turn_command = -MAX_SPEED

print(f"\nAngular Control Command: {turn_command:.2f}")

# --- Interpretation for a robot with Left/Right Motors ---
if abs(turn_command) > 5: # Only turn if the error is significant
    if turn_command > 0:
        # Turn Right: Set Right Motor Speed lower, Left Motor Speed higher
        print(f"ACTION: Turn Right with power proportional to {turn_command:.2f}")
    else:
        # Turn Left: Set Left Motor Speed lower, Right Motor Speed higher
        print(f"ACTION: Turn Left with power proportional to {abs(turn_command):.2f}")
else:
    # Marker is almost centered
    print("ACTION: Stop turning (Marker is centered)")
    

💡 The Physics Behind the Angles

    Yaw (Y-axis): This is the heading of the marker relative to the camera. Use this to steer your system left/right.

    Pitch (X-axis): This is the vertical tilt of the marker. Use this to control a gimbal or an arm moving up/down.

    Roll (Z-axis): This is the twist of the marker in the plane of the image. This is often ignored unless you need a system to specifically match the marker's orientation.

The key is to tune the Kp​ value. If Kp​ is too large, your system will oscillate wildly; if it's too small, it will respond slowly.

create an open cv capture 20 images for callibration script then also give the script for callbration to get the camera matrix and distortion im using 7x9 inner points chessboard 20mmx20mm squares first the capture script

That's a great next step for accurate pose estimation. Camera calibration is essential!

Here are the two Python scripts you need for the 7×9 inner corner chessboard with 20.0mm squares.

📸 1. Image Capture Script

This script will open your camera, display the feed, and save 20 images when you press the SPACEBAR. It's important to capture the board at many different angles and distances for good calibration.

capture_images.py

Python

import cv2
import time
import os

# --- CONFIGURATION ---
NUM_CAPTURES = 20
OUTPUT_DIR = "calibration_images"
# ---------------------

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
print(f"Calibration images will be saved in the '{OUTPUT_DIR}' folder.")

# Open the video stream (0 is typically the default camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Set resolution for better detection (optional, but good practice)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

image_count = 0

while image_count < NUM_CAPTURES:
    ret, frame = cap.read()
    if not ret:
        break

    # Display the current frame
    cv2.putText(frame, f"Captured: {image_count}/{NUM_CAPTURES}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Press SPACE to Capture", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Capture for Calibration", frame)

    key = cv2.waitKey(1) & 0xFF
    
    # Capture image when SPACEBAR is pressed
    if key == ord(' '):
        filename = os.path.join(OUTPUT_DIR, f"calib_img_{image_count:02d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Captured: {filename}")
        image_count += 1
        # Brief pause to avoid capturing multiple images with one key press
        time.sleep(0.5) 
        
    # Exit loop when 'q' is pressed
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nFinished capture. {image_count} images saved.")

📐 2. Calibration Script

This script will load the 20 images, find the 7×9 corners in each, and run the cv2.calibrateCamera function to output your crucial parameters.

calibrate_camera.py

Python

import numpy as np
import cv2
import glob
import os

# --- CHESSBOARD & CAMERA CONFIGURATION ---
# Inner corners: 7 points wide, 9 points tall (7x9)
CHESSBOARD_SIZE = (9, 7) 
# Square size in the unit you want the Tvec to be in (e.g., millimeters)
SQUARE_SIZE_MM = 20.0 
CALIB_IMAGES_DIR = "calibration_images"
CALIB_OUTPUT_FILE = "calib_data.npz"
# ------------------------------------------

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (20,0,0), (40,0,0) ...
# This creates a grid of 3D points representing the chessboard in the "world"
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE_MM

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Get all calibration images
images = glob.glob(os.path.join(CALIB_IMAGES_DIR, '*.jpg'))

if not images:
    print(f"Error: No images found in '{CALIB_IMAGES_DIR}'. Run capture_images.py first.")
    exit()

print(f"Found {len(images)} images. Processing...")

# Image dimensions for calibration (will be set by first image)
img_size = None 

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if img_size is None:
        img_size = gray.shape[::-1] # (width, height)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    # If found, add object points, and refine image points
    if ret:
        objpoints.append(objp)
        
        # Refine corner locations for sub-pixel accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners (Optional visualization)
        img = cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
        
    else:
        print(f"Warning: Could not find corners in {fname}")

    # Display progress
    cv2.imshow('Calibration Progress', img)
    cv2.waitKey(50)

cv2.destroyAllWindows()

if not objpoints:
    print("Error: Calibration failed. No corners were successfully detected in any image.")
    exit()

# --- CALIBRATION ---
print("\nStarting camera calibration...")

# Calibrate the camera
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_size, None, None
)

# --- RESULTS ---
print("\n--- CALIBRATION SUCCESSFUL ---")
print(f"RMS Error: {ret:.3f} (Lower is better)")
print("\nCamera Matrix (Intrinsic Parameters):")
print(cameraMatrix)
print("\nDistortion Coefficients:")
print(distCoeffs)

# Save the necessary data for pose estimation
np.savez(CALIB_OUTPUT_FILE, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)
print(f"\nCalibration data saved to {CALIB_OUTPUT_FILE}")

🔑 Key Outputs

    cameraMatrix: The 3×3 matrix containing the focal lengths (fx​,fy​) and the optical center (cx​,cy​). This is crucial for perspective calculations.

    distCoeffs: The 1×5 or 1×8 vector containing the lens distortion parameters (radial and tangential). You pass this to cv2.undistort or cv2.solvePnP to correct warped images.

us efind Chessboard and draw chessboard corners

Here is a concise Python script that uses cv2.findChessboardCorners to locate the corners of your 9×7 chessboard and then uses cv2.drawChessboardCorners to visualize the detected points.

This code assumes you have an image saved in your current directory, perhaps one of the files captured earlier like "calibration_images/calib_img_00.jpg".
Python

import cv2
import numpy as np

# --- CONFIGURATION (Based on your previous setup) ---
CHESSBOARD_SIZE = (9, 7) # Inner corners: 9 points wide, 7 points tall
IMAGE_PATH = "calibration_images/calib_img_00.jpg" # Replace with your image file
# ----------------------------------------------------

# 1. Load Image and Convert to Grayscale
img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"Error: Could not load image at {IMAGE_PATH}. Check the path.")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Find Chessboard Corners
# 'ret' will be True if corners are found
# 'corners' will contain the (x, y) coordinates of the corners
ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

# 3. Process and Draw if Corners are Found
if ret:
    print("Corners found! Refining and drawing...")
    
    # Define termination criteria for sub-pixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Refine corner locations for higher accuracy
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Draw the detected corners onto the original image
    # Note: We draw on the original 'img' (color) for better visualization
    img_with_corners = cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners_refined, ret)
    
    # Display the result
    cv2.imshow('Detected Chessboard Corners', img_with_corners)
    cv2.waitKey(0)
    
else:
    print("Error: Could not find chessboard corners in the image.")

cv2.destroyAllWindows()

Key Functions Explained:

    cv2.findChessboardCorners(image, patternSize, corners):

        image: The grayscale image (recommended) to search within.

        patternSize: A tuple (cols, rows) specifying the number of inner corners. For a 9×7 pattern, you have 9 corners horizontally and 7 vertically.

        corners: The output array containing the 2D pixel coordinates of the detected corners.

        Returns: A boolean (ret) indicating success, and the corners array.

    cv2.drawChessboardCorners(image, patternSize, corners, patternWasFound):

        image: The image to draw on (usually the color image).

        patternSize: The same inner corner size used in findChessboardCorners.

        corners: The array of corner coordinates (ideally the refined ones).

        patternWasFound: The boolean result from findChessboardCorners (or simply True). The function draws colors indicating if the pattern was successfully found.

only capture if found corners

That makes perfect sense! Only capturing images where the corners are successfully found will greatly improve the quality and success rate of your final calibration.

Here is the modified script. The key change is using cv2.findChessboardCorners right after pressing the spacebar and saving the image only if it returns True.

capture_corners_only.py

Python

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

💡 How the Logic Works

    When you press SPACE, the script converts the frame to grayscale.

    It calls cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None).

    The variable corners_found is a boolean (True or False).

    The script only enters the if corners_found: block if the pattern was successfully detected, ensuring the image is saved only when valid calibration data is present.

    It also draws the corners on the frame using cv2.drawChessboardCorners so you get visual confirmation of a successful capture before it moves on to the next.

no the space bar should only capture when there are corners

The script provided in the previous turn, capture_corners_only.py, already does this.

The logic is contained within the if key == ord(' '): block, and the key action is the conditional check:
Python

        # --- CORE LOGIC: Find Corners ---
        corners_found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        if corners_found:
            # ONLY executes if corners_found is True
            # ... draw corners
            # ... save image (cv2.imwrite)
            # ... increment counter

The image is saved using cv2.imwrite only inside the if corners_found: block.

Here is the exact section of the code responsible for the conditional saving:
Python

# ... (inside the while True loop) ...
    # Check for capture command
    if key == ord(' '):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # This function finds the corners and returns True/False
        corners_found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        if corners_found:
            # THIS CODE ONLY RUNS IF CORNERS ARE FOUND
            # The image is only saved here:
            filename = os.path.join(OUTPUT_DIR, f"calib_img_{image_count:02d}.jpg")
            cv2.imwrite(filename, frame) 
            
            # ... rest of the successful capture logic
            image_count += 1
            # ...
        else:
            # This code runs if corners are NOT found
            message = "FAILED! Move board and try again."
            # The image is NOT saved here

If you are running the script and believe it is saving images without corners, please verify:

    CHESSBOARD_SIZE = (9, 7) is correctly set to the number of inner corners of your board.

    The marker dictionary and the cv2 library are correctly installed (you are using opencv-contrib-python as recommended).

    The board is flat and fully visible in the frame.

load callibration matrix from npz file

You can load the calibration matrix and distortion coefficients from the .npz file generated by your calibration script using the numpy library's load function.

Here is the simple Python snippet to read that data back into your program for use in pose estimation:
Python

import numpy as np

# --- CONFIGURATION ---
CALIB_OUTPUT_FILE = "calib_data.npz"
# ---------------------

try:
    # 1. Load the data from the .npz file
    with np.load(CALIB_OUTPUT_FILE) as data:
        # 2. Extract the saved variables
        camera_matrix = data['cameraMatrix']
        dist_coeffs = data['distCoeffs']
        
    print("✅ Calibration data loaded successfully!")
    print("Camera Matrix Shape:", camera_matrix.shape)
    print("Distortion Coefficients Shape:", dist_coeffs.shape)

    # You can now use these variables (camera_matrix, dist_coeffs) 
    # in your pose estimation script (e.g., in cv2.aruco.estimatePoseSingleMarkers)

except FileNotFoundError:
    print(f"❌ Error: Calibration file '{CALIB_OUTPUT_FILE}' not found.")
    print("Please ensure you have run the calibration script first.")

Explanation

    with np.load(CALIB_OUTPUT_FILE) as data:: This opens the compressed NumPy file (.npz). The file object is assigned to the variable data.

    data['cameraMatrix']: When you saved the data using np.savez(..., cameraMatrix=cameraMatrix, distCoeffs=distCoeffs), NumPy assigned the names as keys. You use these keys to retrieve the arrays.

    The retrieved arrays are assigned to the variables camera_matrix and dist_coeffs, which are now ready to be passed to your ArUco pose functions.

i need to control pinouts in driver.py

To control hardware pinouts from your Python script named driver.py on a Raspberry Pi, you'll need to use a specialized library that can interact with the Pi's General Purpose Input/Output (GPIO) pins. The most common and reliable library for this is RPi.GPIO.

Since you are using a Raspberry Pi 4, here is a breakdown of the process, including installation, basic setup, and control examples.

1. Installation

If you're running a modern version of Raspberry Pi OS, the RPi.GPIO library should be pre-installed. If not, you can install it via the terminal:
Bash

sudo apt update
sudo apt install python3-rpi.gpio

2. Basic Setup and Pin Numbering

Before controlling the pins, you must import the library and choose a numbering scheme.

There are two primary ways to number the pins:

    GPIO.BOARD: Uses the physical numbering of the pins on the 40-pin header. This is easy to understand, but your code might break if you switch to a different model of Pi with a different layout.

    GPIO.BCM: Uses the Broadcom chip's channel numbers (GPIO numbers). This is more robust as it refers to the GPIO channel regardless of its physical location. BCM is the recommended scheme.

Add this to the beginning of your driver.py script:
Python

import RPi.GPIO as GPIO
import time

# Use BCM numbering scheme (Recommended)
GPIO.setmode(GPIO.BCM)

3. Controlling a Digital Pin (Output)

If you want to turn a component (like an LED or a motor driver signal) ON or OFF, you use a digital output pin.

In your driver.py, let's assume you connect an LED to GPIO pin 17 (BCM number).
Python

# Pin definition
LED_PIN = 17 

# 1. Set the pin as an output
GPIO.setup(LED_PIN, GPIO.OUT)

# 2. Control the pin (High is ON, Low is OFF)
def turn_on_led():
    # Set the pin to HIGH (3.3V)
    GPIO.output(LED_PIN, GPIO.HIGH) 

def turn_off_led():
    # Set the pin to LOW (0V)
    GPIO.output(LED_PIN, GPIO.LOW)

# Example usage:
turn_on_led()
time.sleep(1) # Wait 1 second
turn_off_led()

4. Reading a Digital Pin (Input)

If you want to check the status of a button or a sensor, you use a digital input pin.

Let's assume a button is connected to GPIO pin 27.
Python

BUTTON_PIN = 27

# 1. Set the pin as an input
# pull_up_down=GPIO.PUD_UP sets an internal resistor to HIGH, 
# so the pin is normally HIGH and goes LOW when the button is pressed (to GND).
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# 2. Read the pin status
def check_button():
    # Returns 1 (True/HIGH) or 0 (False/LOW)
    input_state = GPIO.input(BUTTON_PIN)
    
    if input_state == GPIO.LOW:
        print("Button Pressed!")
        return True
    else:
        print("Button Released.")
        return False

# Example usage:
check_button()

5. Cleanup (Essential!)

It is critical to reset the GPIO pins when your script finishes to avoid conflicts with other programs. Always include this line at the end of your main execution path:
Python

# Clears all the settings you have set up
GPIO.cleanup() 

# Pin initialization

leftrear_a = Pin(12, Pin.OUT)

leftrear_b = Pin(13, Pin.OUT)

leftfront_a = Pin(8, Pin.OUT)

leftfront_b = Pin(7, Pin.OUT)


rightrear_a = Pin(4, Pin.OUT)

rightrear_b = Pin(5, Pin.OUT)

rightfront_a = Pin(10, Pin.OUT)    

rightfront_b = Pin(11, Pin.OUT)


# User-defined movement functions

def forward(sleep_time=None):

    try:

        leftrear_b.value(1)

        rightrear_a.value(1)

        #leftfront_a.value(1)

        #rightfront_a.value(1)


        if sleep_time:

            time.sleep(sleep_time)

            stop()

    except Exception as e:

        print(f"Error in forward: {e}")


def backward(sleep_time=None):

    try:

        leftrear_a.value(1)

        rightrear_b.value(1)

        #leftfront_b.value(1)

        #rightfront_b.value(1)


        if sleep_time:

            time.sleep(sleep_time)

            stop()

    except Exception as e:

        print(f"Error in backward: {e}")


def right(sleep_time=None):

    try:

        leftrear_b.value(1)

        leftfront_a.value(1)

        #rightrear_b.value(1)

        #rightfront_b.value(1)


        if sleep_time:

            time.sleep(sleep_time)

            stop()

    except Exception as e:

        print(f"Error in right: {e}")


def left(sleep_time=None):

    try:

        leftrear_b.value(1)

        leftfront_b.value(1)

        rightrear_a.value(1)

        rightfront_b.value(1)


        if sleep_time:

            time.sleep(sleep_time)

            stop()

    except Exception as e:

        print(f"Error in left: {e}")


def stop():

    # Stops all motors

    leftrear_a.value(0)

    leftrear_b.value(0)

    leftfront_a.value(0)

    leftfront_b.value(0)

    rightrear_a.value(0)

    rightrear_b.value(0)

    rightfront_a.value(0)

    rightfront_b.value(0)


def main():

    print("Hey, I'm starting")

    forward(2)

    backward(2)

    left(1.5)

    right(1.5)

   

    #right(2)

    stop()

    print("Sequence done")


if __name__ == "__main__":

    main()



see make this into raspberry pi code

That code is written for a microcontroller environment (likely MicroPython, given the Pin.value(1) syntax). To make it run on a Raspberry Pi, you need to replace the Pin class with the appropriate RPi.GPIO functions, as discussed in our previous conversation.

Here is the code, fully converted to use the BCM pin numbering scheme with the standard Python time library and structured for a Raspberry Pi.

🤖 Raspberry Pi Motor Control Script

This script controls a four-wheeled robot chassis using 8 GPIO pins (likely connected to a motor driver like an L298N) for basic movements.

driver.py (Raspberry Pi/RPi.GPIO Version)

Python

import RPi.GPIO as GPIO
import time

# --- CONFIGURATION: Use BCM Pin Numbering ---
# BCM is recommended as it refers to the GPIO channel, not the physical pin number.
GPIO.setmode(GPIO.BCM) 

# --- PIN INITIALIZATION ---
# Assign BCM pin numbers (These are the GPIO numbers, NOT the physical pin numbers)
# NOTE: These pin numbers must match where you physically wired the Raspberry Pi.

# Left Side Motor Driver Input Pins
LEFT_REAR_A = 12 
LEFT_REAR_B = 13
LEFT_FRONT_A = 8
LEFT_FRONT_B = 7

# Right Side Motor Driver Input Pins
RIGHT_REAR_A = 4
RIGHT_REAR_B = 5
RIGHT_FRONT_A = 10
RIGHT_FRONT_B = 11

# Group all pins together
PINS = [
    LEFT_REAR_A, LEFT_REAR_B, LEFT_FRONT_A, LEFT_FRONT_B,
    RIGHT_REAR_A, RIGHT_REAR_B, RIGHT_FRONT_A, RIGHT_FRONT_B
]

# Set all pins as OUTPUT
for pin in PINS:
    GPIO.setup(pin, GPIO.OUT)

# --- USER-DEFINED MOVEMENT FUNCTIONS ---

def forward(sleep_time=None):
    """Moves the robot straight forward."""
    try:
        # Left Side: A=LOW, B=HIGH (or vice versa, depending on wiring)
        # Right Side: A=HIGH, B=LOW
        
        # Original logic: leftrear_b=1, rightrear_a=1
        # To make a 4-wheel drive, let's assume all 'B' pins are LOW and all 'A' pins are HIGH for one direction (or vice versa).
        
        # Turn all motors ON in the forward direction
        GPIO.output(LEFT_REAR_A, GPIO.LOW)
        GPIO.output(LEFT_REAR_B, GPIO.HIGH)
        GPIO.output(LEFT_FRONT_A, GPIO.LOW)
        GPIO.output(LEFT_FRONT_B, GPIO.HIGH)
        
        GPIO.output(RIGHT_REAR_A, GPIO.HIGH) # Right side needs inverse logic
        GPIO.output(RIGHT_REAR_B, GPIO.LOW)
        GPIO.output(RIGHT_FRONT_A, GPIO.HIGH)
        GPIO.output(RIGHT_FRONT_B, GPIO.LOW)

        if sleep_time:
            time.sleep(sleep_time)
            stop()
    except Exception as e:
        print(f"Error in forward: {e}")

def backward(sleep_time=None):
    """Moves the robot straight backward."""
    try:
        # Turn all motors ON in the backward direction
        GPIO.output(LEFT_REAR_A, GPIO.HIGH)
        GPIO.output(LEFT_REAR_B, GPIO.LOW)
        GPIO.output(LEFT_FRONT_A, GPIO.HIGH)
        GPIO.output(LEFT_FRONT_B, GPIO.LOW)
        
        GPIO.output(RIGHT_REAR_A, GPIO.LOW) # Right side needs inverse logic
        GPIO.output(RIGHT_REAR_B, GPIO.HIGH)
        GPIO.output(RIGHT_FRONT_A, GPIO.LOW)
        GPIO.output(RIGHT_FRONT_B, GPIO.HIGH)

        if sleep_time:
            time.sleep(sleep_time)
            stop()
    except Exception as e:
        print(f"Error in backward: {e}")

def right(sleep_time=None):
    """Turns the robot right (by rotating left wheels forward and right wheels backward)."""
    try:
        # Left wheels forward
        GPIO.output(LEFT_REAR_A, GPIO.LOW)
        GPIO.output(LEFT_REAR_B, GPIO.HIGH)
        GPIO.output(LEFT_FRONT_A, GPIO.LOW)
        GPIO.output(LEFT_FRONT_B, GPIO.HIGH)
        
        # Right wheels backward
        GPIO.output(RIGHT_REAR_A, GPIO.LOW)
        GPIO.output(RIGHT_REAR_B, GPIO.HIGH)
        GPIO.output(RIGHT_FRONT_A, GPIO.LOW)
        GPIO.output(RIGHT_FRONT_B, GPIO.HIGH)

        if sleep_time:
            time.sleep(sleep_time)
            stop()
    except Exception as e:
        print(f"Error in right: {e}")

def left(sleep_time=None):
    """Turns the robot left (by rotating right wheels forward and left wheels backward)."""
    try:
        # Left wheels backward
        GPIO.output(LEFT_REAR_A, GPIO.HIGH)
        GPIO.output(LEFT_REAR_B, GPIO.LOW)
        GPIO.output(LEFT_FRONT_A, GPIO.HIGH)
        GPIO.output(LEFT_FRONT_B, GPIO.LOW)
        
        # Right wheels forward
        GPIO.output(RIGHT_REAR_A, GPIO.HIGH)
        GPIO.output(RIGHT_REAR_B, GPIO.LOW)
        GPIO.output(RIGHT_FRONT_A, GPIO.HIGH)
        GPIO.output(RIGHT_FRONT_B, GPIO.LOW)

        if sleep_time:
            time.sleep(sleep_time)
            stop()
    except Exception as e:
        print(f"Error in left: {e}")

def stop():
    """Stops all motors by setting all control pins to LOW."""
    for pin in PINS:
        GPIO.output(pin, GPIO.LOW)

def main():
    try:
        print("Hey, I'm starting the motor sequence.")
        forward(2)
        backward(2)
        left(1.5)
        right(1.5)
        
        stop()
        print("Sequence done.")
        
    except KeyboardInterrupt:
        print("\nSequence interrupted.")
        
    finally:
        # --- CRITICAL: Cleanup GPIO pins ---
        GPIO.cleanup()
        print("GPIO cleanup complete.")

if __name__ == "__main__":
    main()

⚠️ Important Notes for Raspberry Pi Use

1. Library and Setup

    import RPi.GPIO as GPIO: This replaces the Pin class.

    GPIO.setmode(GPIO.BCM): We set the numbering scheme to BCM. You must use the BCM numbers (the GPIO channel numbers), not the physical pin numbers (the BOARD numbers).

    GPIO.output(pin, value): This replaces the pin.value(value) syntax. GPIO.HIGH (or 1) is ON, and GPIO.LOW (or 0) is OFF.

2. Motor Logic

The original functions only toggled one or two pins per movement. For a robust four-wheeled robot, the forward, backward, left, and right functions in the Pi code above were adjusted to control all four motors simultaneously for smoother movement.

3. Safety Cleanup

The finally: GPIO.cleanup() block is essential. It ensures that even if an error occurs or the script is stopped with Ctrl+C, all the GPIO pins are safely reset to their default input state.

import cv2

import numpy as np

import driver


def main():

    dictionary  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)# check later


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


add controls to use the driver to follow this path

    Query successful

That's an excellent challenge! You have all the necessary components: the ArUco pose estimation logic, the Euler angle extraction, and the motor control functions in driver.py.

To make the robot follow the zigzag path shown in the image, we'll use a state machine and PID-like control logic based on the distance (Z-axis) and the angle (Yaw) of the marker.

The path consists of multiple segments, each being 24 cm long and requiring a 30∘ turn.

🎯 Control Strategy

We will break the path into two repeating control objectives:

    Alignment (Turn): When the robot is at a waypoint, it turns (e.g., 30∘ right) until the marker is straight ahead again (Yaw≈0).

    Navigation (Forward): Once aligned, it moves forward until the distance (Z-axis) is reduced by 24 cm.

1. Pin Logic Correction

In your provided driver.py, the right() and left() functions only toggle one side of the robot. For proper turning on a 4-wheel drive platform, you need to drive the wheels on one side forward and the wheels on the other side backward.

I'll use the corrected turning logic from the previous turn for the final script.

2. The Main Control Script

I will rewrite the main() function to include the state machine and control logic.

    Target Distance: 24 cm or 0.24 meters per segment.

    Target Angle: ≈0∘ (straight ahead) after each turn.

    Proportional Gain (Kp​): A tuning factor to manage the turn speed.

Python

import cv2
import numpy as np
import driver
import time

# --- CONSTANTS ---
MARKER_SIZE = 0.05  # 5cm
# The path has 4 segments of 24 cm, requiring 3 turns (30 deg each)
# The segments alternate between Right and Left turns.
SEGMENT_DISTANCE = 0.24  # 24 cm in meters
TURN_ANGLE = 30.0        # Degrees to turn at each waypoint
# Proportional gain for turning control (TUNE THIS!)
KP_YAW = 0.5             
YAW_TOLERANCE = 3.0      # Stop turning if Yaw is within +/- 3 degrees
# --- END CONSTANTS ---

# --- CAMERA SETUP (Using your provided values) ---
camera_matrix = np.array([[600, 0, 320],
                          [0, 600, 240],
                          [0, 0, 1]], dtype=float)
dist_coeffs = np.zeros((4, 1))

# --- ARUCO SETUP (Using your legacy functions) ---
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
parameters = cv2.aruco.DetectorParameters_create()

def get_pose_data(frame, dictionary, parameters, camera_matrix, dist_coeffs, marker_size):
    """Detects marker, estimates pose, and extracts Yaw, Z-distance."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    
    if ids is not None and len(ids) == 1:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
        
        rvec, tvec = rvecs[0], tvecs[0]
        R, _ = cv2.Rodrigues(rvec)
        
        # Yaw Angle calculation (Rotation around Y-axis)
        try:
            # Yaw (rotation around Y, most important for steering)
            yaw_rad = np.arctan2(R[1, 0], R[0, 0])
            yaw_degrees = np.degrees(yaw_rad)
        except ValueError:
            yaw_degrees = 0 # Gimbal lock or error
        
        # Distance (Z-axis)
        z_dist = tvec[0][2]
        
        return z_dist, yaw_degrees, frame
        
    return None, None, frame

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

def main_control():
    # Sequence of turns needed: Right 30, Left 30, Right 30, Left 30 (Based on the zigzag)
    # NOTE: The image shows 4 segments, thus 3 turning points (30, 30, 30).
    # Assuming the robot starts facing the first segment:
    turn_sequence = [TURN_ANGLE, -TURN_ANGLE, TURN_ANGLE] # Right, Left, Right
    
    current_segment = 0
    segments_to_travel = len(turn_sequence) + 1 # 4 segments total
    
    # Initialize camera
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    time.sleep(1)
    if not cap.isOpened():
        print("Failed to open camera.")
        return

    # Check initial pose and set first target distance
    ret, initial_frame = cap.read()
    if not ret: return
    z_dist_start, _, _ = get_pose_data(initial_frame, dictionary, parameters, camera_matrix, dist_coeffs, MARKER_SIZE)
    
    if z_dist_start is None:
        print("Cannot find marker at start. Exiting.")
        driver.stop()
        return

    # --- STATE MACHINE VARIABLES ---
    state = "ALIGNMENT" 
    target_z_dist = z_dist_start - SEGMENT_DISTANCE
    
    try:
        while current_segment < segments_to_travel:
            ret, frame = cap.read()
            if not ret:
                print("Lost camera feed.")
                break

            z_dist, yaw_degrees, frame = get_pose_data(frame, dictionary, parameters, camera_matrix, dist_coeffs, MARKER_SIZE)
            
            if z_dist is None:
                print("Marker lost. Stopping.")
                driver.stop()
                break

            # ----------------------------------------------------
            # 1. ALIGNMENT STATE (Turning)
            # ----------------------------------------------------
            if state == "ALIGNMENT":
                print(f"STATE: ALIGNMENT (Yaw: {yaw_degrees:.1f} deg)")
                
                # Turn Right (Positive Yaw error) or Left (Negative Yaw error) until aligned (Yaw approx 0)
                is_aligned = turn_to_target_yaw(0.0, yaw_degrees) 
                
                if is_aligned:
                    # Once aligned, the turning movement is complete.
                    print(f"ALIGNMENT ACHIEVED for segment {current_segment + 1}.")
                    state = "NAVIGATION"
                    # Set the new target distance after alignment
                    z_dist_start_of_segment = z_dist 
                    target_z_dist = z_dist_start_of_segment - SEGMENT_DISTANCE
            
            # ----------------------------------------------------
            # 2. NAVIGATION STATE (Moving Forward)
            # ----------------------------------------------------
            elif state == "NAVIGATION":
                print(f"STATE: NAVIGATION (Z: {z_dist:.2f}m, Target Z: {target_z_dist:.2f}m)")
                
                if z_dist > target_z_dist:
                    # Move forward while adjusting yaw slightly
                    driver.forward()
                    # Optional: Add small yaw correction here using KP_YAW
                else:
                    # Segment distance reached! Stop and prepare for the next turn.
                    driver.stop()
                    current_segment += 1
                    print(f"SEGMENT {current_segment} complete.")
                    
                    if current_segment < len(turn_sequence):
                        # Prepare for the next segment turn
                        # The robot must physically turn the required angle.
                        turn_angle_to_make = turn_sequence[current_segment] 
                        
                        if turn_angle_to_make > 0:
                            print(f"Turning {TURN_ANGLE} degrees RIGHT.")
                            driver.right(sleep_time=1.0) # Assume 1.0 second turn time for 30 deg
                        else:
                            print(f"Turning {TURN_ANGLE} degrees LEFT.")
                            driver.left(sleep_time=1.0) # Assume 1.0 second turn time for 30 deg

                        state = "ALIGNMENT" # Go back to alignment to correct for the physical turn
                        
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

if __name__ == "__main__":
    driver.main() # Run driver test sequence (optional)
    main_control() # Run the ArUco path follower

⚙️ How to Tune the Code

    Motor Control Speed (Driver): You will need to modify the driver.py functions to include PWM (Pulse Width Modulation) control to manage motor speed, allowing the turn_command to control how fast the robot turns. Right now, it's just ON/OFF.

    Kp​ (Proportional Gain): The line KP_YAW = 0.5 is a critical tuning value.

        If the robot turns too aggressively or overshoots, decrease Kp​.

        If the robot turns too slowly, increase Kp​.

    Turn Time: The physical turn time sleep_time=1.0 is an estimate. You must calibrate how long it takes your robot to physically turn 30∘ at the speed you've set, or implement a more advanced PID controller for the turns themselves.

def main():

    dictionary  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)# check later


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





use this as a base take the parameters taking script

Got it. You want to integrate the loading of the camera calibration data from a file into your main script, replacing the hardcoded dummy values for camera_matrix and dist_coeffs.

This is the correct approach for an operational robotics system. I will modify your main() function to use numpy.load to get the parameters from the file created in the earlier steps.

Here is the updated script, assuming your calibration file is named calib_data.npz (the default name from the calibration script).
Python

import cv2
import numpy as np
import driver
import time
import os

# --- CONFIGURATION ---
CALIB_FILE = "calib_data.npz" # Name of your calibration file
MARKER_SIZE = 0.05             # 5cm (Must match your physical marker size in meters)
# Proportional gain for turning control (TUNE THIS!)
KP_YAW = 0.5             
YAW_TOLERANCE = 3.0      # Stop turning if Yaw is within +/- 3 degrees
SEGMENT_DISTANCE = 0.24  # 24 cm in meters
TURN_ANGLE = 30.0        # Degrees to turn at each waypoint
# --- END CONFIGURATION ---

# --- ARUCO SETUP (Using your legacy functions) ---
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
parameters = cv2.aruco.DetectorParameters_create()

def load_calibration_parameters():
    """Loads camera matrix and distortion coefficients from the .npz file."""
    if not os.path.exists(CALIB_FILE):
        print(f"❌ ERROR: Calibration file '{CALIB_FILE}' not found.")
        print("Please run the calibration script first!")
        # Use dummy values as a fallback to avoid crash, but performance will be poor
        return None, None
    
    try:
        with np.load(CALIB_FILE) as data:
            camera_matrix = data['cameraMatrix']
            dist_coeffs = data['distCoeffs']
        print(f"✅ Calibration data loaded from {CALIB_FILE}.")
        return camera_matrix, dist_coeffs
    except Exception as e:
        print(f"❌ ERROR loading calibration data: {e}")
        return None, None

# --- Pose/Angle Extraction Function ---
def get_pose_data(frame, dictionary, parameters, camera_matrix, dist_coeffs, marker_size):
    """Detects marker, estimates pose, and extracts Yaw, Z-distance."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    
    if ids is not None and len(ids) == 1:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
        
        rvec, tvec = rvecs[0], tvecs[0]
        R, _ = cv2.Rodrigues(rvec)
        
        # Yaw Angle calculation (Rotation around Y-axis)
        try:
            # Yaw (rotation around Y, most important for steering)
            yaw_rad = np.arctan2(R[1, 0], R[0, 0])
            yaw_degrees = np.degrees(yaw_rad)
        except ValueError:
            yaw_degrees = 0
        
        # Distance (Z-axis)
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
    
    if camera_matrix is None:
        # If loading failed, exit the program
        return

    # Initialize camera
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
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
            
            if z_dist is None:
                print("Marker lost. Stopping.")
                driver.stop()
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
            cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
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

if __name__ == "__main__":
    main()


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



# if __name__ == "__main__":
#     main_control()