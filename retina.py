import os
import cv2
import numpy as np
from datetime import datetime
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis

# Initialize InsightFace
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# Load known faces
face_groups = defaultdict(list)
for filename in os.listdir("faces"):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
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
        name = filename.split('_')[0]
        face_groups[name].append(faces[0].embedding)

# Average embeddings
known_face_encodings = []
known_face_names = []
for name, embeddings in face_groups.items():
    avg_embedding = np.mean(embeddings, axis=0)
    known_face_encodings.append(avg_embedding)
    known_face_names.append(name)

# Prepare video input/output
video_path = "output2.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter("outputnew2.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
log_file = open("recognition_log.txt", "w")

# Folder to save cropped faces
os.makedirs("faces_cropped", exist_ok=True)

# Process video
frame_number = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    for face in faces:
        bbox = face.bbox.astype(int)
        embedding = face.embedding
        name = "Unknown"

        # Recognize
        if known_face_encodings:
            similarities = cosine_similarity([embedding], known_face_encodings)[0]
            max_idx = np.argmax(similarities)
            if similarities[max_idx] > 0.15:
                name = known_face_names[max_idx]
                log_file.write(f"{timestamp} - {name}\n")
                print(f"{timestamp} - {name}")

        # Draw bounding box and label on the video frame
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Save padded cropped face (without drawing on it)
        h, w, _ = frame.shape
        pad_w = int(1.2 * (x2 - x1))
        pad_h = int(1.2 * (y2 - y1))
        x1_pad = max(x1 - pad_w, 0)
        y1_pad = max(y1 - pad_h, 0)
        x2_pad = min(x2 + pad_w, w)
        y2_pad = min(y2 + pad_h, h)

        cropped_face = frame[y1_pad:y2_pad, x1_pad:x2_pad].copy()
        if cropped_face.size == 0:
            continue

        cropped_face = cv2.resize(cropped_face, (160, 160))
        person_dir = os.path.join("faces_cropped", name)
        os.makedirs(person_dir, exist_ok=True)
        filename = os.path.join(person_dir, f"{timestamp}_{frame_number}.jpg")
        cv2.imwrite(filename, cropped_face)

    out.write(frame)
    frame_number += 1
    cv2.imshow("Video Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
out.release()
log_file.close()
cv2.destroyAllWindows()
