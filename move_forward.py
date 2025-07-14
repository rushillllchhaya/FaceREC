import os
import cv2
import numpy as np
from datetime import datetime
from collections import defaultdict, deque
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
from deep_sort_realtime.deepsort_tracker import DeepSort

# Helper functions
def compute_iou(boxA, boxB):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

def box_center_distance(boxA, boxB):
    """Calculate Euclidean distance between box centers"""
    ax, ay = (boxA[0] + boxA[2]) / 2, (boxA[1] + boxA[3]) / 2
    bx, by = (boxB[0] + boxB[2]) / 2, (boxB[1] + boxB[3]) / 2
    return np.linalg.norm([ax - bx, ay - by])

# Initialize InsightFace with stricter detection thresholds
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_thresh=0.4)  # Increased detection threshold

# Load known faces with quality check
face_groups = defaultdict(list)
MIN_FACE_SIZE = 80  # Minimum face size in pixels to consider

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
        
        # Check face size
        face = faces[0]
        bbox = face.bbox.astype(int)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
            print(f"⚠️ Face too small in: {filename} ({width}x{height}px)")
            continue
            
        print(f"✅ Face detected in: {filename} ({width}x{height}px)")
        name = filename.split('_')[0]
        face_groups[name].append(face.embedding)

# Average embeddings only if we have multiple samples
known_face_encodings = []
known_face_names = []
for name, embeddings in face_groups.items():
    if len(embeddings) > 1:
        avg_embedding = np.mean(embeddings, axis=0)
    else:
        avg_embedding = embeddings[0]
    known_face_encodings.append(avg_embedding)
    known_face_names.append(name)

# Initialize video capture
video_path = "output2.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("Could not open video file")

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output_improved.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
log_file = open("recognition_log.txt", "w")

# Initialize tracker with stricter parameters
tracker = DeepSort(
    max_age=15,
    n_init=3,
    nn_budget=100,
    max_cosine_distance=0.3,  # Stricter matching
    max_iou_distance=0.7
)

# Recognition thresholds
DIRECT_MATCH_THRESHOLD = 0.22 # Increased from 0.21
FALLBACK_MATCH_THRESHOLD = 0.15 # Increased from 0.15   
MIN_CONFIDENCE = 0.28  # Minimum confidence to assign a name

# Persistent mappings with confidence scores
track_id_to_info = {}  # {track_id: (name, confidence, last_update_frame)}
known_track_records = {}  # {track_id: (name, bbox, embedding)}
track_history = defaultdict(lambda: deque(maxlen=30))  # {track_id: [(bbox, embedding)]}

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
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        # Skip small detections
        if width < 50 or height < 50:
            continue
            
        embedding = face.embedding
        detections.append(([bbox[0], bbox[1], width, height], 1.0, 'face'))
        embeddings.append(embedding)
        bboxes.append(bbox)

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        current_bbox = np.array([x1, y1, x2, y2])
        
        # Match detection to track
        matched_idx = -1
        if embeddings:
            # Find closest detection
            centers = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in bboxes])
            track_center = np.array([(x1+x2)/2, (y1+y2)/2])
            distances = np.linalg.norm(centers - track_center, axis=1)
            matched_idx = np.argmin(distances) if len(distances) > 0 else -1

        current_embedding = embeddings[matched_idx] if matched_idx != -1 else None
        
        # Initialize track info
        if track_id not in track_id_to_info:
            track_id_to_info[track_id] = ("Unknown", 0.0, frame_number)
        
        current_name, current_conf, last_update = track_id_to_info[track_id]
        
        # Direct matching with known faces
        if current_embedding is not None and known_face_encodings:
            similarities = cosine_similarity([current_embedding], known_face_encodings)[0]
            max_idx = np.argmax(similarities)
            max_sim = similarities[max_idx]
            
            if max_sim > DIRECT_MATCH_THRESHOLD:
                proposed_name = known_face_names[max_idx]
                
                # Only update if we're more confident than before
                if max_sim > current_conf:
                    track_id_to_info[track_id] = (proposed_name, max_sim, frame_number)
                    known_track_records[track_id] = (proposed_name, current_bbox, current_embedding)
                    log_file.write(f"{timestamp} - ID {track_id}: {proposed_name} (Conf: {max_sim:.2f}, Source: Direct)\n")
                    print(f"{timestamp} - ID {track_id}: {proposed_name} (Conf: {max_sim:.2f}, Source: Direct)")

        # Fallback matching only if we're still uncertain
        if track_id_to_info[track_id][0] == "Unknown" and current_embedding is not None:
            # Check against known tracks with spatial and appearance consistency
            best_match = None
            best_score = 0
            
            for prev_tid, (prev_name, prev_bbox, prev_embedding) in known_track_records.items():
                if prev_tid == track_id:
                    continue
                    
                # Spatial checks
                iou = compute_iou(current_bbox, prev_bbox)
                center_dist = box_center_distance(current_bbox, prev_bbox)
                
                # Appearance similarity
                sim = cosine_similarity([current_embedding], [prev_embedding])[0][0]
                
                # Combined score (weighted towards appearance)
                score = (0.7 * sim) + (0.2 * (1 - center_dist/200)) + (0.1 * iou)
                
                if score > best_score and score > 0.4:  # Higher combined threshold
                    best_score = score
                    best_match = (prev_name, sim)
            
            if best_match and best_match[1] > FALLBACK_MATCH_THRESHOLD:
                proposed_name, match_conf = best_match
                if match_conf > current_conf:
                    track_id_to_info[track_id] = (proposed_name, match_conf, frame_number)
                    known_track_records[track_id] = (proposed_name, current_bbox, current_embedding)
                    log_file.write(f"{timestamp} - ID {track_id}: {proposed_name} (Conf: {match_conf:.2f}, Source: Fallback)\n")
                    print(f"{timestamp} - ID {track_id}: {proposed_name} (Conf: {match_conf:.2f}, Source: Fallback)")

        # Get current name after all matching attempts
        current_name, current_conf, _ = track_id_to_info[track_id]
        
        # Visualize with confidence
        color = (0, 255, 0) if current_name != "Unknown" else (0, 0, 255)
        thickness = 2 if current_conf > MIN_CONFIDENCE else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        label = f"ID {track_id}: {current_name}"
        if current_name != "Unknown":
            label += f" ({current_conf:.2f})"
            
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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