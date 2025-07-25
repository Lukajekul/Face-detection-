import mediapipe as mp
from deepface import DeepFace
import cv2
import threading
import queue

shared_detection = queue.Queue()
shared_image = queue.Queue()
shared_face = queue.Queue()

def face_tracking():
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image_rgb)

            image.flags.writeable = True

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    ih, iw, _ = image.shape
                    x = int(bbox.xmin * iw)
                    y = int(bbox.ymin * ih)
                    w = int(bbox.width * iw)
                    h = int(bbox.height * ih)

                    # Draw box
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # If emotion available, draw text
                    if not shared_face.empty():
                        feeling = shared_face.get()
                        cv2.putText(image, feeling, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Share data for analysis
                    shared_detection.put(detection)
                    shared_image.put(image.copy())

            # Show flipped image
            cv2.imshow('Face Detection + Emotion', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

def head_emotion():
    while True:
        detection = shared_detection.get()
        image = shared_image.get()

        bbox = detection.location_data.relative_bounding_box
        ih, iw, _ = image.shape
        x = int(bbox.xmin * iw)
        y = int(bbox.ymin * ih)
        w = int(bbox.width * iw)
        h = int(bbox.height * ih)

        # Crop and analyze
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(iw, x + w), min(ih, y + h)
        cropped_face = image[y1:y2, x1:x2]

        try:
            cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            details = DeepFace.analyze(cropped_face_rgb, actions=["emotion"], enforce_detection=False)
            feeling = details["dominant_emotion"]
        except:
            feeling = "Unknown"

        shared_face.put(feeling)

# Start threads
t1 = threading.Thread(target=face_tracking)
t2 = threading.Thread(target=head_emotion)

t1.daemon = True
t2.daemon = True

t1.start()
t2.start()

t1.join()
