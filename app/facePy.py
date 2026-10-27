#Arkansas Bot Face Recognition System
import cv2
from insightface.app import FaceAnalysis
from cv2_enumerate_cameras import enumerate_cameras
import mediapipe as mp
import numpy as np
import os
import time
import platform

#Validation for the platform to use the correct backend for camera capture
if platform.system() == "Darwin":      # Mac
    backend = cv2.CAP_AVFOUNDATION     # Platform-specific backend for macOS
else:                                  # Jetson / Linux
    backend = cv2.CAP_V4L2             # Platform-specific backend for Linux

# Initialize MediaPipe Hands once outside the function to prevent memory leaks
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

#Hand detection function using MediaPipe    
def hand_detection(frame):
    # Convert the frame to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    # Draw hand landmarks on the frame if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return frame

#Face recognition function using InsightFace
def face_recognition(frame, face_app):
    # Convert the frame to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform face detection and recognition
    faces = face_app.get(rgb_frame)

    # Draw bounding boxes and landmarks for detected faces
    for face in faces:
        bbox = face.bbox.astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        #Validation to prevent errors if Landmark is None
        if face.landmark is not None:
            for landmark in face.landmark:
                cv2.circle(frame, (int(landmark[0]), int(landmark[1])), 2, (0, 0, 255), -1)

    return frame

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
    face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Cannot read frame")
            break
        frame = hand_detection(frame)
        frame = face_recognition(frame, face_app)
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): #Close the camera window when 'q' is pressed
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()