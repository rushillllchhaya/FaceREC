import os
import cv2
import face_recognition
from mtcnn import MTCNN

# Initialize MTCNN detector
detector = MTCNN()

# Load known faces
known_face_encodings = []
known_face_names = []

for filename in os.listdir("faces"):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join("faces", filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb_frame)

    face_locations = []

    for det in detections:
        x, y, width, height = det["box"]
        top = max(0, y)
        right = x + width
        bottom = y + height
        left = max(0, x)
        face_locations.append((top, right, bottom, left))

    # Use the full image and detected face locations
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.45)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

        # Draw bounding box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Face Recognition with MTCNN", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
