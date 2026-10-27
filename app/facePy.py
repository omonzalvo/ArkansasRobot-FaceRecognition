#Arkansas Bot Face Recognition System
import cv2
from insightface.app import FaceAnalysis
from cv2_enumerate_cameras import enumerate_cameras
import numpy as np
import os
import time
import platform

#Validation for the platform to use the correct backend for camera capture
if platform.system() == "Darwin":      # Mac
    backend = cv2.CAP_AVFOUNDATION
else:                                  # Jetson / Linux
    backend = cv2.CAP_V4L2

def main():
    cameras = enumerate_cameras(backend)

    if not cameras:
        print("No camera found")
        return

    for opcion, info in enumerate(cameras):
        print(f"{opcion}: {info.name}")

    election = int(input("Choose your camera: "))

    cap = cv2.VideoCapture(cameras[election].index, backend)

    # Initialize the FaceAnalysis app
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Cannot read frame")
            break
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): #Close the camera window when 'q' is pressed
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()