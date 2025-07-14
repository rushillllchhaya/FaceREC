import os
import cv2
import numpy as np
from datetime import datetime
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

# CONFIG
CAMERA_URL = "http://media.arcisai.io:8080/fmp4/DVR/RTSP-ATPL-910178-AIPTZ.mp4"  
# CAMERA_URL = "http://media.arcisai.io:8080/fmp4/DVR/RTSP-ATPL-910419-AIPTZ.mp4"  # CHANGE THIS TO YOUR CAMERA LINK
# CAMERA_URL = "http://media.arcisai.io:8080/fmp4/DVR/RTSP-ATPL-908612-AIPTZ.mp4" # CHANGE THIS TO YOUR CAMERA LINK
MAX_RETRIES = 3
RETRY_DELAY = 3  # seconds

# Thresholds for matching
INITIAL_RECOGNITION_THRESHOLD = 0.35   # Primary recognition threshold
PROXIMITY_FEATURE_THRESHOLD = 0.30    # Lower threshold for nearby faces
PROXIMITY_DISTANCE = 50               # Maximum distance to consider faces as "close"
MIN_SIMILARITY_FOR_COPY = 0.50        # Minimum similarity to copy identity

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

# Initialize tracker
tracker = DeepSort(max_age=10, n_init=2)
track_id_to_name = {}
known_track_records = {}

# Prepare output
os.makedirs("faces_cropped", exist_ok=True)
log_file = open("recognition_log.txt", "w")

# Helper functions
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

def find_nearby_known_faces(current_bbox, current_embedding, tracks, max_distance=PROXIMITY_DISTANCE):
    """Find nearby faces that are already recognized and check facial similarity"""
    nearby_matches = []
    
    for other_track in tracks:
        if not other_track.is_confirmed():
            continue
            
        other_track_id = other_track.track_id
        if other_track_id not in track_id_to_name:
            continue
            
        other_name = track_id_to_name[other_track_id]
        if other_name == "Unknown":
            continue
            
        other_bbox = list(map(int, other_track.to_ltrb()))
        distance = box_center_distance(current_bbox, other_bbox)
        
        if distance <= max_distance:
            # Check if we have stored embedding for this track
            if other_track_id in known_track_records:
                _, _, other_embedding = known_track_records[other_track_id]
                if other_embedding is not None and current_embedding is not None:
                    # Calculate facial similarity
                    similarity = cosine_similarity([current_embedding], [other_embedding])[0][0]
                    nearby_matches.append((other_name, distance, similarity, other_track_id))
    
    return nearby_matches

# Open video stream
def open_camera(url):
    return cv2.VideoCapture(url)

cap = open_camera(CAMERA_URL)
retry_count = 0
frame_number = 0

while True:
    if not cap.isOpened():
        print("⚠️ Camera not opened. Reconnecting...")
        cap.release()
        time.sleep(RETRY_DELAY)
        cap = open_camera(CAMERA_URL)
        continue

    ret, frame = cap.read()

    if not ret:
        print(f"⚠️ Frame not received. Retrying ({retry_count + 1}/{MAX_RETRIES})...")
        retry_count += 1
        time.sleep(RETRY_DELAY)
        if retry_count >= MAX_RETRIES: 
            print("❌ Max retries reached. Reconnecting camera...")
            cap.release()
            cap = open_camera(CAMERA_URL)
            retry_count = 0
        continue
    
    frame = cv2.resize(frame, (1720, 870))  # Resize to 1920x1080
    retry_count = 0  # reset on success

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    faces = app.get(frame)

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

        # Find the best matching detection for this track
        min_dist = float('inf')
        matched_idx = -1
        for i, bbox in enumerate(bboxes):
            dist = box_center_distance([x1, y1, x2, y2], bbox)
            if dist < 200 and dist < min_dist:
                matched_idx = i
                min_dist = dist

        if matched_idx == -1:
            continue

        embedding = embeddings[matched_idx]

        # Check if this track already has a name assigned
        if track_id in track_id_to_name:
            name = track_id_to_name[track_id]
        else:
            # Primary recognition against known faces database
            if known_face_encodings and embedding is not None:
                similarities = cosine_similarity([embedding], known_face_encodings)[0]
                max_idx = np.argmax(similarities)
                if similarities[max_idx] > INITIAL_RECOGNITION_THRESHOLD:
                    name = known_face_names[max_idx]
                    track_id_to_name[track_id] = name
                    known_track_records[track_id] = (name, np.array([x1, y1, x2, y2]), embedding)
                    log_file.write(f"{timestamp} - ID {track_id}: {name} (Primary Recognition - Similarity: {similarities[max_idx]:.3f})\n")
                    print(f"{timestamp} - ID {track_id}: {name} (Primary Recognition - Similarity: {similarities[max_idx]:.3f})")

            # If still unknown, check against historical track records
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

            # Enhanced proximity-based matching with facial feature comparison
            if name == "Unknown" and embedding is not None:
                nearby_matches = find_nearby_known_faces([x1, y1, x2, y2], embedding, tracks)
                
                if nearby_matches:
                    # Sort by a combination of facial similarity and proximity
                    # Higher weight on facial similarity, lower weight on distance
                    scored_matches = []
                    for match_name, distance, similarity, other_track_id in nearby_matches:
                        if similarity > MIN_SIMILARITY_FOR_COPY:
                            # Combined score: high similarity is good, low distance is good
                            combined_score = similarity * 0.7 + (1 / (distance + 1)) * 0.3
                            scored_matches.append((match_name, combined_score, similarity, distance, other_track_id))
                    
                    if scored_matches:
                        # Get the best match
                        best_match = max(scored_matches, key=lambda x: x[1])
                        best_name, best_score, best_similarity, best_distance, best_other_track_id = best_match
                        
                        # Additional check: ensure similarity is reasonable
                        if best_similarity > MIN_SIMILARITY_FOR_COPY:
                            name = best_name
                            track_id_to_name[track_id] = name
                            known_track_records[track_id] = (name, np.array([x1, y1, x2, y2]), embedding)
                            log_file.write(f"{timestamp} - ID {track_id}: {name} (Proximity+Feature Match - Similarity: {best_similarity:.3f}, Distance: {best_distance:.1f})\n")
                            print(f"{timestamp} - ID {track_id}: {name} (Proximity+Feature Match - Similarity: {best_similarity:.3f}, Distance: {best_distance:.1f})")

            # Fallback: IoU-based matching (keep existing logic as last resort)
            if name == "Unknown":
                for other_track_id, other_name in track_id_to_name.items():
                    if other_track_id == track_id:
                        continue
                    other_track = next((t for t in tracks if t.track_id == other_track_id), None)
                    if other_track:
                        iou = compute_iou([x1, y1, x2, y2], list(map(int, other_track.to_ltrb())))
                        if iou > 0.2:
                            name = other_name
                            track_id_to_name[track_id] = name
                            known_track_records[track_id] = (name, np.array([x1, y1, x2, y2]), embedding)
                            log_file.write(f"{timestamp} - ID {track_id}: {name} (IoU Match - IoU: {iou:.3f})\n")
                            print(f"{timestamp} - ID {track_id}: {name} (IoU Match - IoU: {iou:.3f})")
                            break

        # Draw bounding box and label
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {track_id}: {name}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Uncomment if you want to save cropped faces
        # h, w, _ = frame.shape
        # pad_w = int(1.2 * (x2 - x1))
        # pad_h = int(1.2 * (y2 - y1))
        # x1_pad = max(x1 - pad_w, 0)
        # y1_pad = max(y1 - pad_h, 0)
        # x2_pad = min(x2 + pad_w, w)
        # y2_pad = min(y2 + pad_h, h)

        # cropped_face = frame[y1_pad:y2_pad, x1_pad:x2_pad].copy()
        # if cropped_face.size == 0:
        #     continue

        # cropped_face = cv2.resize(cropped_face, (160, 160))
        # person_dir = os.path.join("faces_cropped", name)
        # os.makedirs(person_dir, exist_ok=True)
        # filename = os.path.join(person_dir, f"{timestamp}_{frame_number}.jpg")
        # cv2.imwrite(filename, cropped_face)

    cv2.imshow("Video Recognition", frame)
    frame_number += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
log_file.close()
cv2.destroyAllWindows()