import mediapipe as mp
from deepface import DeepFace
import cv2
import threading
import queue
import time


shared_detection = queue.Queue()
shared_image = queue.Queue()

def face_tracking():
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)

            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)
                    
                    shared_detection.put(detection)
                    shared_image.put(image.copy())









            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))


            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

def head_emotion():
    while True:
        detection = shared_detection.get()
        image= shared_image.get()
        bbox = detection.location_data.relative_bounding_box
        ih, iw, _ = image.shape
        x = int(bbox.xmin * iw)
        y = int(bbox.ymin * ih)
        w = int(bbox.width * iw)
        h = int(bbox.height * ih)

        # Crop the face from the frame
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(iw, x + w), min(ih, y + h)
        cropped_face = image[y1:y2, x1:x2]

        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        details = DeepFace.analyze(cropped_face, actions=["age", "gender", "race", "emotion"], enforce_detection=False)

        print(details)





t1 = threading.Thread(target=face_tracking)
t2 = threading.Thread(target=head_emotion)

t1.deamon = True
t2.deamon = True

t1.start()
t2.start()