import cv2
import os
import numpy as np
import insightface
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis

# Initialize InsightFace
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# Load known faces with logging
known_face_encodings = []
known_face_names = []

for filename in os.listdir("faces"):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join("faces", filename)
        img = cv2.imread(path)

        if img is None:
            print(f"❌ Could not read image: {filename}")
            continue

        faces = app.get(img)

        if not faces:
            print(f"⚠️ No face detected in: {filename}")
            continue

        print(f"✅ Face detected in: {filename}")
        known_face_encodings.append(faces[0].embedding)
        known_face_names.append(os.path.splitext(filename)[0])


# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)
    for face in faces:
        bbox = face.bbox.astype(int)
        embedding = face.embedding
        name = "Unknown"

        # Compare with known embeddings
        if known_face_encodings:

            similarities = cosine_similarity([embedding], known_face_encodings)[0]
            max_idx = np.argmax(similarities)
            if similarities[max_idx] > 0.5:  # Threshold (higher = more strict)
                name = known_face_names[max_idx]


        # Draw bounding box
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("InsightFace Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
