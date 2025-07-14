import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
import cv2
import numpy as np
import os
import pickle
from datetime import datetime
import time
from collections import defaultdict, deque
import threading
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import onnxruntime as ort
from ultralytics import YOLO
import faiss
from sklearn.metrics.pairwise import cosine_similarity

class AdvancedCCTVFaceRecognition:
    def __init__(self, 
                 known_faces_dir="known_faces",
                 confidence_threshold=0.7,
                 face_threshold=0.4,
                 use_gpu=True,
                 model_type="insightface"):
        """
        Advanced CCTV Face Recognition with state-of-the-art models
        
        Args:
            known_faces_dir: Directory containing known face images
            confidence_threshold: Minimum confidence for face recognition
            face_threshold: Minimum threshold for face detection
            use_gpu: Whether to use GPU acceleration
            model_type: 'insightface', 'arcface', or 'facenet'
        """
        self.known_faces_dir = known_faces_dir
        self.confidence_threshold = confidence_threshold
        self.face_threshold = face_threshold
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_history = deque(maxlen=30)
        
        # Face tracking
        self.face_tracker = {}
        self.next_face_id = 0
        self.track_history = defaultdict(list)
        
        # Initialize models
        self.init_models()
        
        # Face database
        self.face_database = None
        self.known_names = []
        self.load_face_database()
        
    def init_models(self):
        """Initialize advanced face detection and recognition models"""
        print(f"Initializing models on {self.device}...")
        
        # 1. YOLOv8 Face Detection (Superior to HOG)
        try:
            self.face_detector = YOLO('yolov8n-face.pt')  # You can use yolov8s-face.pt for better accuracy
            print("✓ YOLOv8 Face Detector loaded")
        except:
            print("! YOLOv8 face model not found, using alternative...")
            self.face_detector = None
        
        # 2. InsightFace for Face Recognition (ArcFace backbone)
        if self.model_type == "insightface":
            try:
                self.face_model = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                self.face_model.prepare(ctx_id=0 if self.device.type == 'cuda' else -1, det_size=(640, 640))
                print("✓ InsightFace (ArcFace) model loaded")
            except Exception as e:
                print(f"! InsightFace initialization failed: {e}")
                self.init_fallback_model()
        
        # 3. Face Quality Assessment Model
        self.init_quality_model()
        
        # 4. Face Anti-Spoofing Model
        self.init_antispoofing_model()
        
    def init_fallback_model(self):
        """Initialize fallback models if advanced models fail"""
        print("Initializing fallback FaceNet model...")
        # Load pre-trained FaceNet model
        try:
            from facenet_pytorch import MTCNN, InceptionResnetV1
            self.mtcnn = MTCNN(keep_all=True, device=self.device)
            self.facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            print("✓ FaceNet model loaded as fallback")
        except:
            print("! All advanced models failed, using basic face_recognition library")
            import face_recognition
            self.use_basic_model = True
    
    def init_quality_model(self):
        """Initialize face quality assessment model"""
        # Simple quality assessment based on blur detection and face size
        self.quality_threshold = 0.3
        
    def init_antispoofing_model(self):
        """Initialize anti-spoofing model to detect fake faces"""
        # Placeholder for anti-spoofing model
        # In production, you'd load models like Silent-Face-Anti-Spoofing
        pass
    
    def detect_faces_yolo(self, frame):
        """Detect faces using YOLOv8"""
        if self.face_detector is None:
            return []
        
        results = self.face_detector(frame, conf=self.face_threshold)
        faces = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    faces.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'landmarks': None  # YOLOv8 doesn't provide landmarks by default
                    })
        
        return faces
    
    def detect_faces_insightface(self, frame):
        """Detect faces using InsightFace"""
        try:
            faces = self.face_model.get(frame)
            face_data = []
            
            for face in faces:
                bbox = face.bbox.astype(int)
                face_data.append({
                    'bbox': [bbox[0], bbox[1], bbox[2], bbox[3]],
                    'confidence': float(face.det_score),
                    'landmarks': face.kps,
                    'embedding': face.embedding,
                    'age': getattr(face, 'age', None),
                    'gender': getattr(face, 'sex', None)
                })
            
            return face_data
        except:
            return []
    
    def assess_face_quality(self, face_crop):
        """Assess the quality of detected face"""
        # Calculate blur metric using Laplacian variance
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate face size
        h, w = face_crop.shape[:2]
        size_score = min(h, w) / 100.0  # Normalize to 0-1
        
        # Combined quality score
        quality_score = (blur_score / 1000.0 + size_score) / 2
        
        return min(quality_score, 1.0)
    
    def extract_face_embedding(self, face_crop, face_data=None):
        """Extract high-quality face embedding"""
        if self.model_type == "insightface" and face_data and 'embedding' in face_data:
            return face_data['embedding']
        
        # Fallback to FaceNet or other models
        try:
            # Preprocess image
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            face_tensor = transform(face_crop).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.facenet_model(face_tensor).cpu().numpy().flatten()
            
            return embedding
        except:
            # Ultimate fallback
            import face_recognition
            rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_face)
            return encodings[0] if encodings else None
    
    def load_face_database(self):
        """Load and create face database with advanced indexing"""
        print("Loading face database...")
        
        # Check for existing database in faces folder
        database_path = os.path.join("faces", "advanced_face_db.pkl")
        
        if os.path.exists(database_path):
            with open(database_path, 'rb') as f:
                data = pickle.load(f)
                self.face_database = data['embeddings']
                self.known_names = data['names']
                
                # Print loading summary
                if 'person_counts' in data:
                    person_counts = data['person_counts']
                    print(f"Loaded face database:")
                    print(f"  Total embeddings: {len(self.known_names)}")
                    print(f"  Unique persons: {len(person_counts)}")
                    for person, count in person_counts.items():
                        print(f"    {person}: {count} embeddings")
                else:
                    print(f"Loaded {len(self.known_names)} faces from database")
        else:
            print("No existing database found. Creating new database...")
            self.create_face_database()
        
        # Create FAISS index for fast similarity search
        if self.face_database is not None and len(self.face_database) > 0:
            dimension = len(self.face_database[0])
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            embeddings_array = np.array(self.face_database).astype('float32')
            faiss.normalize_L2(embeddings_array)
            self.faiss_index.add(embeddings_array)
            print("✓ FAISS index created for fast face search")
        else:
            print("⚠ No face database loaded - please check your faces folder")
    
    def create_face_database(self):
        """Create face database from faces directory with multiple images per person"""
        faces_dir = "faces"  # Your existing faces folder
        if not os.path.exists(faces_dir):
            print(f"Faces directory '{faces_dir}' not found!")
            print("Please ensure your faces folder exists with person subfolders")
            return
        
        embeddings = []
        names = []
        person_embeddings = {}  # Store multiple embeddings per person
        
        print("Creating face database from 'faces' folder...")
        print("Processing multiple images per person for better recognition...")
        
        # Check if faces folder contains subfolders (person names) or direct images
        items = os.listdir(faces_dir)
        has_subfolders = any(os.path.isdir(os.path.join(faces_dir, item)) for item in items)
        
        if has_subfolders:
            # Process person subfolders (recommended structure)
            print("Detected person subfolders structure")
            for person_name in os.listdir(faces_dir):
                person_path = os.path.join(faces_dir, person_name)
                if os.path.isdir(person_path):
                    person_embeddings[person_name] = []
                    processed_count = 0
                    
                    print(f"\nProcessing {person_name}:")
                    for filename in os.listdir(person_path):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_path = os.path.join(person_path, filename)
                            embedding = self.process_single_image(image_path, person_name, filename)
                            
                            if embedding is not None:
                                person_embeddings[person_name].append(embedding)
                                processed_count += 1
                    
                    print(f"  ✓ Processed {processed_count} images for {person_name}")
        else:
            # Process images directly in faces folder (fallback method)
            print("Processing images directly from faces folder")
            print("Note: For better organization, create subfolders for each person")
            
            # Group images by person name (assuming format: personname_01.jpg, personname_02.jpg)
            for filename in os.listdir(faces_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Extract person name (everything before last underscore or the whole filename)
                    base_name = os.path.splitext(filename)[0]
                    if '_' in base_name:
                        person_name = '_'.join(base_name.split('_')[:-1])  # Remove last part after underscore
                    else:
                        person_name = base_name
                    
                    if person_name not in person_embeddings:
                        person_embeddings[person_name] = []
                    
                    image_path = os.path.join(faces_dir, filename)
                    embedding = self.process_single_image(image_path, person_name, filename)
                    
                    if embedding is not None:
                        person_embeddings[person_name].append(embedding)
        
        # Create final database with averaged embeddings or multiple embeddings per person
        print(f"\nFinal processing:")
        for person_name, person_embs in person_embeddings.items():
            if len(person_embs) > 0:
                if len(person_embs) == 1:
                    # Single embedding for this person
                    embeddings.append(person_embs[0])
                    names.append(person_name)
                    print(f"  {person_name}: 1 embedding")
                else:
                    # Multiple embeddings - we can either average them or keep them separate
                    # Option 1: Keep all embeddings (better for recognition)
                    for i, emb in enumerate(person_embs):
                        embeddings.append(emb)
                        names.append(person_name)  # Same name for all embeddings
                    print(f"  {person_name}: {len(person_embs)} embeddings")
                    
                    # Option 2: Also create an averaged embedding (uncomment if needed)
                    # avg_embedding = np.mean(person_embs, axis=0)
                    # embeddings.append(avg_embedding)
                    # names.append(f"{person_name}_avg")
        
        self.face_database = embeddings
        self.known_names = names
        
        # Save database
        database_path = os.path.join("faces", "advanced_face_db.pkl")
        with open(database_path, 'wb') as f:
            pickle.dump({
                'embeddings': embeddings,
                'names': names,
                'person_counts': {name: names.count(name) for name in set(names)}
            }, f)
        
        # Print summary
        unique_persons = set(names)
        print(f"\n✓ Face database created!")
        print(f"  Total embeddings: {len(embeddings)}")
        print(f"  Unique persons: {len(unique_persons)}")
        for person in unique_persons:
            count = names.count(person)
            print(f"    {person}: {count} embeddings")
    
    def process_single_image(self, image_path, person_name, filename):
        """Process a single image and return embedding"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"    ✗ Could not load: {filename}")
            return None
        
        # Detect faces in the image
        faces = self.detect_faces_insightface(image)
        
        if not faces:
            print(f"    ✗ No face detected in: {filename}")
            return None
        
        # Use the largest face (most likely the main subject)
        face = max(faces, key=lambda f: (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1]))
        
        bbox = face['bbox']
        face_crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        
        # Assess quality
        quality = self.assess_face_quality(face_crop)
        if quality < self.quality_threshold:
            print(f"    ⚠ Low quality ({quality:.2f}): {filename}")
            # Don't skip low quality images entirely, just warn
        
        # Extract embedding
        embedding = self.extract_face_embedding(face_crop, face)
        
        if embedding is not None:
            print(f"    ✓ {filename} (Quality: {quality:.2f})")
            return embedding
        else:
            print(f"    ✗ Failed to extract embedding: {filename}")
            return None
    
    def recognize_face(self, face_embedding):
        """Recognize face using advanced similarity search with multiple embeddings per person"""
        if self.face_database is None or len(self.face_database) == 0:
            return "Unknown", 0.0
        
        try:
            # Use FAISS for fast search - get top 5 matches to handle multiple embeddings per person
            query_embedding = np.array([face_embedding]).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            k = min(5, len(self.face_database))  # Get top 5 matches or all if less than 5
            distances, indices = self.faiss_index.search(query_embedding, k=k)
            
            if len(distances[0]) > 0:
                # Analyze top matches to find best person match
                person_scores = {}
                
                for i in range(len(distances[0])):
                    similarity = distances[0][i]
                    match_idx = indices[0][i]
                    person_name = self.known_names[match_idx]
                    
                    # Keep track of best score for each person
                    if person_name not in person_scores:
                        person_scores[person_name] = similarity
                    else:
                        person_scores[person_name] = max(person_scores[person_name], similarity)
                
                # Find the person with highest score
                if person_scores:
                    best_person = max(person_scores, key=person_scores.get)
                    best_score = person_scores[best_person]
                    
                    if best_score > self.confidence_threshold:
                        return best_person, float(best_score)
        
        except Exception as e:
            print(f"FAISS search failed: {e}, using fallback...")
            # Fallback to cosine similarity
            similarities = cosine_similarity([face_embedding], self.face_database)[0]
            
            # Group by person and find best match for each person
            person_scores = {}
            for i, similarity in enumerate(similarities):
                person_name = self.known_names[i]
                if person_name not in person_scores:
                    person_scores[person_name] = similarity
                else:
                    person_scores[person_name] = max(person_scores[person_name], similarity)
            
            if person_scores:
                best_person = max(person_scores, key=person_scores.get)
                best_score = person_scores[best_person]
                
                if best_score > self.confidence_threshold:
                    return best_person, float(best_score)
        
        return "Unknown", 0.0
    
    def track_faces(self, faces, frame_shape):
        """Advanced face tracking across frames"""
        # Simple centroid-based tracking
        current_centroids = []
        for face in faces:
            bbox = face['bbox']
            centroid = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            current_centroids.append(centroid)
        
        # Update tracking (simplified version)
        for i, centroid in enumerate(current_centroids):
            faces[i]['track_id'] = i  # Simplified tracking ID
        
        return faces
    
    def process_frame(self, frame):
        """Process frame with advanced face recognition"""
        # Detect faces
        faces = self.detect_faces_insightface(frame)
        
        # Track faces
        faces = self.track_faces(faces, frame.shape)
        
        # Process each face
        results = []
        for face in faces:
            bbox = face['bbox']
            face_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            
            # Quality assessment
            quality = self.assess_face_quality(face_crop)
            
            if quality < self.quality_threshold:
                continue
            
            # Extract embedding and recognize
            embedding = self.extract_face_embedding(face_crop, face)
            if embedding is not None:
                name, confidence = self.recognize_face(embedding)
                
                results.append({
                    'bbox': bbox,
                    'name': name,
                    'confidence': confidence,
                    'quality': quality,
                    'track_id': face.get('track_id', 0),
                    'age': face.get('age'),
                    'gender': face.get('gender')
                })
        
        return results
    
    def draw_advanced_results(self, frame, results):
        """Draw advanced visualization with additional info"""
        for result in results:
            bbox = result['bbox']
            name = result['name']
            confidence = result['confidence']
            quality = result['quality']
            
            # Color coding
            if name == "Unknown":
                color = (0, 0, 255)  # Red
            elif confidence > 0.72:
                color = (0, 255, 0)  # Green - High confidence  
            else:
                color = (255, 0, 0)  # Blue - Medium confidence
            
            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Prepare label with additional info
            label = f"{name}"
            if name != "Unknown":
                label += f" ({confidence:.2f})"
            
            # Add quality and additional info
            info_text = f"Q:{quality:.2f}"
            if result.get('age'):
                info_text += f" Age:{result['age']:.0f}"
            if result.get('gender'):
                gender = "M" if result['gender'] == 1 else "F"
                info_text += f" {gender}"
            
            # Draw labels
            cv2.rectangle(frame, (bbox[0], bbox[1]-35), (bbox[2], bbox[1]), color, -1)
            cv2.putText(frame, label, (bbox[0]+5, bbox[1]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, info_text, (bbox[0]+5, bbox[1]-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame

    def setup_camera_stream(self, camera_source):
        """Setup camera stream with proper configuration for CCTV cameras"""
        print(f"Connecting to camera: {camera_source}")
        
        # Create VideoCapture object
        cap = cv2.VideoCapture()
        
        # Configure for network streams
        if isinstance(camera_source, str) and ('rtsp://' in camera_source or 'http://' in camera_source):
            print("Detected network camera stream")
            # Optimize for network streams
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
            cap.set(cv2.CAP_PROP_FPS, 25)        # Typical CCTV frame rate
            
            # For RTSP streams
            if 'rtsp://' in camera_source:
                # Use TCP for more reliable connection (optional)
                # You might need to modify the URL to include TCP transport
                pass
        
        # Open the camera
        if not cap.open(camera_source):
            print(f"Error: Could not connect to camera source: {camera_source}")
            return None
        
        # Test if we can read a frame first
        ret, test_frame = cap.read()
        if not ret:
            print("Error: Could not read from camera stream")
            cap.release()
            return None
        
        # Get camera's native resolution
        native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Force target resolution (always resize to this)
        target_width, target_height = 960, 540
        
        print(f"⚠ Camera native resolution: {native_width}x{native_height}")
        print(f"✓ Will force resize all frames to {target_width}x{target_height}")
        
        # Always force resize since network cameras rarely accept resolution changes
        self.force_resize = True
        self.target_resolution = (target_width, target_height)
        
        # Store native resolution for reference
        final_width, final_height = native_width, native_height
        
        print(f"✓ Camera connected successfully")
        print(f"  Native Resolution: {final_width}x{final_height}")
        print(f"  Target Resolution: {target_width}x{target_height}")
        print(f"  FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        
        return cap

    def read_frame(self, cap):
        """Read frame from camera and ensure it's the correct resolution"""
        ret, frame = cap.read()
        
        if ret and hasattr(self, 'force_resize') and self.force_resize:
            # Resize frame to target resolution
            frame = cv2.resize(frame, self.target_resolution, interpolation=cv2.INTER_LINEAR)
        
        return ret, frame

    def run_advanced_recognition(self, camera_source=0):
        """Run advanced face recognition system with network camera support"""
        print("Starting Advanced CCTV Face Recognition System...")
        print("Features: YOLOv8 Detection + InsightFace Recognition + Quality Assessment")
        print("Press 'q' to quit, 'a' to add new face, 's' to save frame, 'r' to reconnect")
        
        # Setup camera with proper configuration
        cap = self.setup_camera_stream(camera_source)
        if cap is None:
            return
        
        # Connection monitoring
        consecutive_failures = 0
        max_failures = 10
        
        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                print(f"Frame read failed ({consecutive_failures}/{max_failures})")
                
                if consecutive_failures >= max_failures:
                    print("Too many consecutive failures. Attempting to reconnect...")
                    cap.release()
                    time.sleep(2)  # Wait before reconnecting
                    cap = self.setup_camera_stream(camera_source)
                    if cap is None:
                        print("Reconnection failed. Exiting...")
                        break
                    consecutive_failures = 0
                    continue
                else:
                    time.sleep(0.1)  # Brief pause before retry
                    continue
            
            consecutive_failures = 0  # Reset on successful frame read
            
            start_time = time.time()
            
            # Process frame
            results = self.process_frame(frame)
            
            # Draw results
            frame = self.draw_advanced_results(frame, results)
            
            # Calculate and display FPS
            process_time = time.time() - start_time
            fps = 1.0 / process_time if process_time > 0 else 0
            self.fps_history.append(fps)
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            
            # Display system info
            info_text = [
                f"FPS: {avg_fps:.1f}",
                f"Faces: {len(results)}",
                f"Device: {self.device}",
                f"Model: {self.model_type}"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(frame, text, (10, 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Log high-confidence detections
            for result in results:
                if result['name'] != "Unknown" and result['confidence'] > 0.8:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{timestamp}] {result['name']} detected (Conf: {result['confidence']:.3f}, Q: {result['quality']:.2f})")
            
            cv2.imshow('Advanced CCTV Face Recognition', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"capture_{timestamp}.jpg", frame)
                print(f"Frame saved as capture_{timestamp}.jpg")
            elif key == ord('r'):
                print("Manual reconnection requested...")
                cap.release()
                cap = self.setup_camera_stream(camera_source)
                if cap is None:
                    print("Manual reconnection failed. Exiting...")
                    break
                consecutive_failures = 0
        
        cap.release()
        cv2.destroyAllWindows()

# Usage Examples for Different Camera Types
if __name__ == "__main__":
    # Initialize the advanced system
    advanced_system = AdvancedCCTVFaceRecognition(
        known_faces_dir="known_faces",
        confidence_threshold=0.7,  # Higher threshold for better accuracy
        face_threshold=0.4,        # Face detection threshold
        use_gpu=True,              # Use GPU if available
        model_type="insightface"   # Use InsightFace for best results
    )
    
    # ===== CAMERA CONNECTION OPTIONS =====
    
    rtsp_url = "http://media.arcisai.io:8080/fmp4/DVR/RTSP-ATPL-910172-AIPTZ.mp4"

    # REPLACE THIS LINE WITH YOUR CAMERA URL
    camera_source = input("Enter your camera URL (or press Enter for webcam): ").strip()
    if not camera_source:
        camera_source = 0  # Default to webcam
    
    # Run the system
    advanced_system.run_advanced_recognition(camera_source)