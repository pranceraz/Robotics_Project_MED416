import cv2



def main():
    dictionary  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    
    detector = cv2.aruco.ArucoDetector(dictionary= dictionary,detectorParams= parameters)
    while True:
        cap = cv2.VideoCapture(0)
        ret,frame  = cap.read()
        if not ret:
            print("WHYYYYYYYYYYY")
            break
        print(f"{frame.shape} is the shape")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners,ids,rejected = detector.detectMarkers(frame)
        if ids is not None:
            # This function draws the bounding box and the ID on the frame
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        #<<display frame
        cv2.imshow('Aruco Detector',frame)
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
