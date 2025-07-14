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
# CAMERA_URL = "http://media.arcisai.io:8080/fmp4/DVR/RTSP-ATPL-910178-AIPTZ.mp4"  
# CAMERA_URL = "http://media.arcisai.io:8080/fmp4/DVR/RTSP-ATPL-910419-AIPTZ.mp4"  # CHANGE THIS TO YOUR CAMERA LINK
CAMERA_URL = "http://media.arcisai.io:8080/fmp4/DVR/RTSP-ATPL-908612-AIPTZ.mp4" # CHANGE THIS TO YOUR CAMERA LINK
MAX_RETRIES = 3
RETRY_DELAY = 3  # seconds

# Thresholds for matching
INITIAL_RECOGNITION_THRESHOLD = 0.35  # Primary recognition threshold
PROXIMITY_FEATURE_THRESHOLD = 0.25    # Lower threshold for nearby faces
PROXIMITY_DISTANCE = 150              # Maximum distance to consider faces as "close"
MIN_SIMILARITY_FOR_COPY = 0.5        # Minimum similarity to copy identity

# Frame skipping settings
PROCESS_EVERY_N_FRAMES = 1            # Process every 2nd frame (skip 1 frame)
PROXIMITY_CHECK_INTERVAL = 5          # Check proximity matching every N processed frames
TRACK_MEMORY_FRAMES = 5              # Remember track positions for N frames

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

# Average embeddings and precompute
known_face_encodings = []
known_face_names = []
for name, embeddings in face_groups.items():
    avg_embedding = np.mean(embeddings, axis=0)
    known_face_encodings.append(avg_embedding)
    known_face_names.append(name)

# Convert to numpy array for faster cosine similarity computation
if known_face_encodings:
    known_face_encodings = np.array(known_face_encodings)

# Initialize tracker
tracker = DeepSort(max_age=15, n_init=3)  # Increased max_age for frame skipping
track_id_to_name = {}
known_track_records = {}
track_last_positions = {}
last_processed_frame = None

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
    return np.sqrt((ax - bx)**2 + (ay - by)**2)

def find_nearby_known_faces_optimized(current_bbox, current_embedding, processed_frame_count, max_distance=PROXIMITY_DISTANCE):
    """Optimized version with frame skipping considerations"""
    nearby_matches = []
    
    # Only do expensive proximity checks periodically
    if processed_frame_count % PROXIMITY_CHECK_INTERVAL != 0:
        return nearby_matches
    
    current_center = ((current_bbox[0] + current_bbox[2]) / 2, (current_bbox[1] + current_bbox[3]) / 2)
    
    for other_track_id, other_name in track_id_to_name.items():
        if other_name == "Unknown":
            continue
            
        # Use cached position if available
        if other_track_id in track_last_positions:
            other_bbox = track_last_positions[other_track_id]
            other_center = ((other_bbox[0] + other_bbox[2]) / 2, (other_bbox[1] + other_bbox[3]) / 2)
            distance = np.sqrt((current_center[0] - other_center[0])**2 + (current_center[1] - other_center[1])**2)
            
            if distance <= max_distance and other_track_id in known_track_records:
                _, _, other_embedding = known_track_records[other_track_id]
                if other_embedding is not None and current_embedding is not None:
                    # Fast cosine similarity
                    similarity = np.dot(current_embedding, other_embedding) / (np.linalg.norm(current_embedding) * np.linalg.norm(other_embedding))
                    nearby_matches.append((other_name, distance, similarity, other_track_id))
    
    return nearby_matches

def interpolate_tracks(current_tracks, last_frame_tracks):
    """Simple interpolation for skipped frames"""
    if last_frame_tracks is None:
        return current_tracks
    
    # Update positions for existing tracks
    for track in current_tracks:
        if track.is_confirmed():
            track_id = track.track_id
            if track_id in track_last_positions:
                # Update cached position
                track_last_positions[track_id] = list(map(int, track.to_ltrb()))
    
    return current_tracks

# def save_cropped_face(frame, bbox, name, track_id, timestamp):
#     """Save cropped face image"""
#     x1, y1, x2, y2 = bbox
#     # Add padding to the crop
#     padding = 10
#     x1 = max(0, x1 - padding)
#     y1 = max(0, y1 - padding)
#     x2 = min(frame.shape[1], x2 + padding)
#     y2 = min(frame.shape[0], y2 + padding)
    
#     cropped_face = frame[y1:y2, x1:x2]
#     if cropped_face.size > 0:
#         filename = f"faces_cropped/{name}_{track_id}_{timestamp}.jpg"
#         cv2.imwrite(filename, cropped_face)

def cleanup_old_records():
    """Clean up old track records to prevent memory buildup"""
    global known_track_records, track_last_positions
    if len(known_track_records) > 100:  # Keep only recent 100 records
        # Remove oldest records (simple approach)
        keys_to_remove = list(known_track_records.keys())[:20]
        for key in keys_to_remove:
            if key in known_track_records:
                del known_track_records[key]
            if key in track_last_positions:
                del track_last_positions[key]

# Open video stream
def open_camera(url):
    cap = cv2.VideoCapture(url)
    # Set buffer size to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

cap = open_camera(CAMERA_URL)
retry_count = 0
frame_number = 0
processed_frame_count = 0
last_tracks = None

print("🎥 Starting face recognition system...")
print(f"📊 Loaded {len(known_face_names)} known faces: {', '.join(known_face_names)}")

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
    
    frame = cv2.resize(frame, (1720, 870))
    retry_count = 0
    frame_number += 1

    # Frame skipping logic
    if frame_number % PROCESS_EVERY_N_FRAMES != 0:
        # Display last processed frame or current frame
        display_frame = last_processed_frame if last_processed_frame is not None else frame.copy()
        
        # Draw existing tracks on skipped frames
        if last_tracks is not None:
            for track in last_tracks:
                if track.is_confirmed():
                    track_id = track.track_id
                    if track_id in track_id_to_name:
                        name = track_id_to_name[track_id]
                        ltrb = track.to_ltrb()
                        x1, y1, x2, y2 = map(int, ltrb)
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(display_frame, f"ID {track_id}: {name}", (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        cv2.imshow("Video Recognition", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # Process this frame
    processed_frame_count += 1
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # Face detection
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

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)
    tracks = interpolate_tracks(tracks, last_tracks)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        name = "Unknown"
        embedding = None

        # Cache track position
        track_last_positions[track_id] = [x1, y1, x2, y2]

        # Find the best matching detection for this track
        min_dist = float('inf')
        matched_idx = -1
        for i, bbox in enumerate(bboxes):
            dist = box_center_distance([x1, y1, x2, y2], bbox)
            if dist < 200 and dist < min_dist:
                matched_idx = i
                min_dist = dist

        if matched_idx == -1:
            # Use existing name if available
            if track_id in track_id_to_name:
                name = track_id_to_name[track_id]
        else:
            embedding = embeddings[matched_idx]

            # Check if this track already has a name assigned
            if track_id in track_id_to_name:
                name = track_id_to_name[track_id]
            else:
                # Primary recognition against known faces database
                if len(known_face_encodings) > 0 and embedding is not None:
                    # Faster cosine similarity using numpy operations
                    similarities = np.dot(known_face_encodings, embedding) / (np.linalg.norm(known_face_encodings, axis=1) * np.linalg.norm(embedding))
                    max_idx = np.argmax(similarities)
                    if similarities[max_idx] > INITIAL_RECOGNITION_THRESHOLD:
                        name = known_face_names[max_idx]
                        track_id_to_name[track_id] = name
                        known_track_records[track_id] = (name, np.array([x1, y1, x2, y2]), embedding)
                        log_file.write(f"{timestamp} - ID {track_id}: {name} (Primary Recognition - Similarity: {similarities[max_idx]:.3f})\n")
                        log_file.flush()
                        print(f"{timestamp} - ID {track_id}: {name} (Primary Recognition - Similarity: {similarities[max_idx]:.3f})")

                # If still unknown, check against historical track records
                if name == "Unknown" and embedding is not None:
                    best_score = 0
                    best_name = None
                    current_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    
                    for prev_tid, (prev_name, prev_bbox, prev_embedding) in known_track_records.items():
                        prev_center = ((prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2)
                        dist = np.sqrt((current_center[0] - prev_center[0])**2 + (current_center[1] - prev_center[1])**2)
                        
                        if dist < 60:
                            sim = np.dot(embedding, prev_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(prev_embedding))
                            score = (1 / (dist + 1e-5)) * sim
                            if score > best_score and sim > 0.10:
                                best_score = score
                                best_name = prev_name
                                
                    if best_name:
                        name = best_name
                        track_id_to_name[track_id] = name
                        known_track_records[track_id] = (name, np.array([x1, y1, x2, y2]), embedding)
                        log_file.write(f"{timestamp} - ID {track_id}: {name} (Historical Track Match - Score: {best_score:.3f})\n")
                        log_file.flush()
                        print(f"{timestamp} - ID {track_id}: {name} (Historical Track Match)")

                # Enhanced proximity-based matching
                if name == "Unknown" and embedding is not None:
                    nearby_matches = find_nearby_known_faces_optimized([x1, y1, x2, y2], embedding, processed_frame_count)
                    
                    if nearby_matches:
                        scored_matches = []
                        for match_name, distance, similarity, other_track_id in nearby_matches:
                            if similarity > MIN_SIMILARITY_FOR_COPY:
                                combined_score = similarity * 0.7 + (1 / (distance + 1)) * 0.3
                                scored_matches.append((match_name, combined_score, similarity, distance, other_track_id))
                        
                        if scored_matches:
                            best_match = max(scored_matches, key=lambda x: x[1])
                            best_name, best_score, best_similarity, best_distance, best_other_track_id = best_match
                            
                            if best_similarity > MIN_SIMILARITY_FOR_COPY:
                                name = best_name
                                track_id_to_name[track_id] = name
                                known_track_records[track_id] = (name, np.array([x1, y1, x2, y2]), embedding)
                                log_file.write(f"{timestamp} - ID {track_id}: {name} (Proximity Match - Similarity: {best_similarity:.3f}, Distance: {best_distance:.1f})\n")
                                log_file.flush()
                                print(f"{timestamp} - ID {track_id}: {name} (Proximity Match - Similarity: {best_similarity:.3f})")

        # # Save cropped face if it's a known person
        # if name != "Unknown" and matched_idx != -1:
        #     save_cropped_face(frame, [x1, y1, x2, y2], name, track_id, timestamp)

        # Draw bounding box and label
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Add confidence info for known faces
        label = f"ID {track_id}: {name}"
        if name != "Unknown" and track_id in known_track_records:
            label += f" ✓"
        
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Store processed frame for skipped frame display
    last_processed_frame = frame.copy()
    last_tracks = tracks

    # Add frame info
    cv2.putText(frame, f"Frame: {frame_number} | Processed: {processed_frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Active Tracks: {len([t for t in tracks if t.is_confirmed()])}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display frame
    cv2.imshow("Video Recognition", frame)
    
    # Cleanup periodically
    if processed_frame_count % 100 == 0:
        cleanup_old_records()
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
log_file.close()
print("🏁 Face recognition system stopped.")
print(f"📝 Log saved to: recognition_log.txt")
print(f"🖼️ Cropped faces saved to: faces_cropped/")