# FaceREC - Face Recognition and Tracking System

This repository contains several Python scripts for face recognition and tracking, primarily utilizing the InsightFace library for robust face detection and embedding extraction, and DeepSORT for multi-object tracking. The project aims to provide various approaches to real-time face recognition in video streams, including advanced features like quality assessment, anti-spoofing (placeholder), and optimized tracking.

## Project Structure

- `clude.py`: Advanced CCTV Face Recognition System with YOLOv8, InsightFace, and FAISS.
- `face_main.py`: Basic face recognition using `face_recognition` library and MTCNN for detection.
- `frameskip.py`: Face recognition and tracking with frame skipping optimization and enhanced proximity matching.
- `insight.py`: Real-time face recognition using InsightFace and OpenCV.
- `move_forward.py`: Improved face recognition and tracking in video files with stricter parameters and consistency checks.
- `retface.py`: Face recognition and tracking in video files with DeepSORT and InsightFace, including saving cropped faces.
- `retface_fr.py`: Enhanced face recognition and tracking with DeepSORT, InsightFace, and advanced proximity-based matching.
- `retina.py`: Basic face recognition in video files using InsightFace, with face cropping and logging.

## File Descriptions




### `clude.py`

This script implements an **Advanced CCTV Face Recognition System**. It integrates:
- **YOLOv8** for robust face detection.
- **InsightFace (ArcFace)** for high-quality face recognition and embedding extraction.
- **FAISS** for fast similarity search in large face databases.
- **Face Quality Assessment** (blur detection, size check).
- **Anti-Spoofing** (placeholder for future integration).
- **Advanced Face Tracking** (centroid-based).
- **CCTV Camera Stream Setup** with optimized network stream handling and forced resolution resizing.
- Real-time FPS display and logging of high-confidence detections.

It supports loading known faces from a `faces` directory, handling multiple images per person, and creating a persistent face database. The system is designed for continuous operation with reconnection logic for camera failures.




### `face_main.py`

This script provides a **basic real-time face recognition system** using a webcam. It leverages:
- **MTCNN** (Multi-task Cascaded Convolutional Networks) for face detection.
- The `face_recognition` library for face encoding and comparison.

It loads known faces from a `faces` directory and then processes live video from a webcam to detect and recognize faces, drawing bounding boxes and names on the frame.




### `frameskip.py`

This script focuses on **optimized face recognition and tracking for video streams**, particularly for CCTV scenarios, by implementing frame skipping. Key features include:
- **InsightFace** for face detection and embedding.
- **DeepSORT** for robust multi-object tracking.
- **Frame Skipping Logic**: Processes only every Nth frame to reduce computational load while maintaining tracking continuity.
- **Enhanced Proximity Matching**: Uses spatial and feature similarity to re-identify faces, especially when they are close to previously known tracks.
- **Dynamic Reconnection**: Handles camera disconnections and attempts to reconnect.
- Logging of recognition events and saving of cropped faces.




### `insight.py`

This script demonstrates **real-time face recognition using InsightFace and OpenCV**. It's a simpler implementation compared to `clude.py` or `frameskip.py`, focusing on:
- **InsightFace** for direct face detection and embedding extraction from a live webcam feed.
- **Cosine Similarity** for comparing detected face embeddings against a database of known faces.

It loads known faces from the `faces` directory and displays the recognized names and bounding boxes on the live video stream.




### `move_forward.py`

This script provides an **improved face recognition and tracking system for video files**. It builds upon previous versions with:
- **Stricter InsightFace detection thresholds** and minimum face size filtering for better quality detections.
- **DeepSORT** with refined parameters (`max_cosine_distance`, `max_iou_distance`) for more accurate tracking.
- **Enhanced matching logic**: Combines direct matching against known faces with fallback mechanisms that consider spatial consistency (IoU, center distance) and appearance similarity for re-identification.
- Persistent tracking information with confidence scores.
- Outputs processed video to `output_improved.mp4` and logs recognition events.




### `retface.py`

This script performs **face recognition and tracking on a video file** (`output2.mp4`) using InsightFace and DeepSORT. Key functionalities include:
- **InsightFace** for face detection and embedding.
- **DeepSORT** for tracking multiple faces across frames.
- **Dynamic Name Assignment**: Assigns names to tracks based on initial recognition against known faces, historical track records, and IoU-based matching for continuity.
- **Cropped Face Saving**: Saves detected and recognized faces to a `faces_cropped` directory, organized by person.
- Logs recognition events to `recognition_log.txt` and outputs processed video to `outputnew6.mp4`.




### `retface_fr.py`

This script is an **enhanced version of face recognition and tracking**, similar to `retface.py` but with more refined matching strategies, particularly for handling challenging scenarios in video streams. It features:
- **InsightFace** for detection and embedding.
- **DeepSORT** for tracking.
- **Advanced Proximity-Based Matching**: Incorporates both spatial distance and facial similarity to link unknown faces to nearby known tracks, improving re-identification.
- **Robust Reconnection Logic**: Handles camera disconnections gracefully.
- Logs recognition events and includes options for saving cropped faces.




### `retina.py`

This script provides a **basic face recognition system for video files** (`output2.mp4`) using InsightFace. It focuses on:
- **InsightFace** for face detection and embedding extraction.
- **Simple Recognition**: Compares detected face embeddings against known faces using cosine similarity.
- **Cropped Face Saving**: Saves detected faces, padded and resized, to a `faces_cropped` directory, organized by recognized name.
- Logs recognized names to `recognition_log.txt` and outputs processed video to `outputnew2.mp4`.
