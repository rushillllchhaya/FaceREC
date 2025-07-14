import os
import cv2
import numpy as np
from datetime import datetime
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
from deep_sort_realtime.deepsort_tracker import DeepSort

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
out = cv2.VideoWriter("outputnew6.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
log_file = open("recognition_log.txt", "w")

# Folder to save cropped faces
os.makedirs("faces_cropped", exist_ok=True)

# Initialize Deep SORT tracker
tracker = DeepSort(max_age=10, n_init=2)

# Persistent mappings
track_id_to_name = {}
known_track_records = {}  # track_id -> (name, bbox, embedding)

# IOU and distance helpers
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

def box_center_distance(boxA, boxB):
    ax, ay = (boxA[0] + boxA[2]) / 2, (boxA[1] + boxA[3]) / 2
    bx, by = (boxB[0] + boxB[2]) / 2, (boxB[1] + boxB[3]) / 2
    return np.linalg.norm([ax - bx, ay - by])

# Process video
frame_number = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    detections = []
    embeddings = []
    bboxes = []

    for face in faces:
        bbox = face.bbox.astype(int)
        embedding = face.embedding
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        detections.append(([x1, y1, width, height], 1.0, 'face'))
        embeddings.append(embedding)
        bboxes.append(bbox)

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        name = "Unknown"
        embedding = None

        # Match detection to track by proximity
        min_dist = float('inf')
        matched_idx = -1
        for i, bbox in enumerate(bboxes):
            bx1, by1, bx2, by2 = bbox
            center_det = np.array([(bx1 + bx2) / 2, (by1 + by2) / 2])
            center_track = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            dist = np.linalg.norm(center_track - center_det)
            if dist < 200 and dist < min_dist:
                matched_idx = i
                min_dist = dist

        if matched_idx == -1:
            continue

        embedding = embeddings[matched_idx]

        if track_id in track_id_to_name:
            name = track_id_to_name[track_id]
        else:
            # Try cosine similarity with known faces
            if known_face_encodings and embedding is not None:
                similarities = cosine_similarity([embedding], known_face_encodings)[0]
                max_idx = np.argmax(similarities)
                if similarities[max_idx] > 0.21:
                    name = known_face_names[max_idx]
                    track_id_to_name[track_id] = name
                    known_track_records[track_id] = (name, np.array([x1, y1, x2, y2]), embedding)
                    log_file.write(f"{timestamp} - ID {track_id}: {name}\n")
                    print(f"{timestamp} - ID {track_id}: {name}")

            # If still unknown, try fallback to previously named track
            if name == "Unknown" and embedding is not None:
                best_score = 0
                best_name = None
                for prev_tid, (prev_name, prev_bbox, prev_embedding) in known_track_records.items():
                    dist = box_center_distance([x1, y1, x2, y2], prev_bbox)
                    if dist < 60:
                        sim = cosine_similarity([embedding], [prev_embedding])[0][0]
                        score = (1 / (dist + 1e-5)) * sim
                        if score > best_score and sim > 0.10:
                            best_score = score
                            best_name = prev_name
                if best_name:
                    name = best_name
                    track_id_to_name[track_id] = name
                    known_track_records[track_id] = (name, np.array([x1, y1, x2, y2]), embedding)

            # 👇 Final fallback: if overlaps with a named track in current frame, reuse name
            if name == "Unknown":
                for other_track_id, other_name in track_id_to_name.items():
                    if other_track_id == track_id:
                        continue
                    other_track = next((t for t in tracks if t.track_id == other_track_id), None)
                    if other_track:
                        other_ltrb = other_track.to_ltrb()
                        iou = compute_iou([x1, y1, x2, y2], list(map(int, other_ltrb)))
                        if iou > 0.2:
                            name = other_name
                            track_id_to_name[track_id] = name
                            known_track_records[track_id] = (name, np.array([x1, y1, x2, y2]), embedding)
                            break

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}: {name}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Save padded cropped face
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
