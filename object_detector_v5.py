import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
import os

class YOLOv5ObjectDetector:
    def __init__(self, model_name='yolov5s', confidence_threshold=0.25, iou_threshold=0.45):
        """
        Optimized Object Detector using YOLOv5 via Ultralytics
        
        Args:
            model_name: YOLOv5 model variant ('yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x')
            confidence_threshold: Minimum confidence for detection
            iou_threshold: IOU threshold for Non-max suppression
        """
        print(f"🚀 Initializing YOLOv5 ({model_name}) Detector...")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🎮 Using Device: {self.device.upper()}")
        
        try:
            # Load the YOLOv5 model
            # This will automatically download weights if not present
            self.model = YOLO(f"{model_name}.pt")
            self.model.to(self.device)
            
            self.confidence_threshold = confidence_threshold
            self.iou_threshold = iou_threshold
            
            # Get class names from the model
            self.classes = self.model.names
            print(f"✅ Loaded {len(self.classes)} object classes")
            
        except Exception as e:
            print(f"❌ Error loading YOLOv5: {e}")
            raise
        
        # Performance tracking
        self.fps_history = []
        self.detection_count = 0
        
        print("✨ Detector ready!\n")
    
    def detect_objects(self, frame):
        """
        Detect objects in frame with YOLOv5
        
        Returns:
            annotated_frame, detections_info
        """
        start_time = time.time()
        
        # Run inference on a batch if needed, but for webcam we do frame by frame
        # We can also skip frames to save CPU/GPU if the framerate is too high
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            half=(self.device == 'cuda'), # Use Half precision on GPU for 2x speed
            imgsz=640, # Standard YOLOv5 size, can be reduced to 320 for speed
            verbose=False
        )
        
        # Ultralytics results[0] contains the detections for our single image
        result = results[0]
        
        # Get annotated frame (Ultralytics provides a handy plot() method)
        annotated_frame = result.plot()
        
        # Process detections for info
        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = self.classes[class_id]
            confidence = float(box.conf[0])
            coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
            
            detections.append({
                'class': label,
                'confidence': confidence,
                'box': coords
            })
        
        # Calculate FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        avg_fps = np.mean(self.fps_history)
        
        # Overlay custom FPS info if needed (Ultralytics plot doesn't include original FPS tracking)
        cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        self.detection_count += len(detections)
        
        return annotated_frame, detections
    
    def run_webcam(self, camera_index=0, width=1280, height=720):
        """Run real-time detection on webcam"""
        print(f"🎥 Starting webcam (camera {camera_index})...")
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not cap.isOpened():
            print("❌ Error: Cannot open camera!")
            return

        print("📋 Press 'q' to quit, 's' to save screenshot")
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            annotated_frame, detections = self.detect_objects(frame)
            cv2.imshow('YOLOv5 Object Detection', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('s'):
                fname = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(fname, annotated_frame)
                print(f"📸 Saved: {fname}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def run_video(self, video_path):
        """Process a video file and save output"""
        print(f"🎬 Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Error: Cannot open video!")
            return
            
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        os.makedirs('output', exist_ok=True)
        output_path = f"output/yolov5_{os.path.basename(video_path)}"
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            annotated_frame, _ = self.detect_objects(frame)
            out.write(annotated_frame)
            cv2.imshow('Processing...', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"✅ Video saved to {output_path}")

if __name__ == "__main__":
    # Use yolov5s (small) for a good balance of speed and accuracy
    # Alternatives: yolov5n (nano), yolov5m (medium), yolov5l (large), yolov5x (extra large)
    detector = YOLOv5ObjectDetector(model_name='yolov5s')
    
    print("\n1. Webcam\n2. Video Path\n3. Image Path")
    choice = input("Select mode: ")
    
    if choice == '1':
        detector.run_webcam()
    elif choice == '2':
        path = input("Path: ")
        detector.run_video(path)
    elif choice == '3':
        path = input("Path: ")
        img = cv2.imread(path)
        if img is not None:
            res_img, dets = detector.detect_objects(img)
            cv2.imshow('Result', res_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Invalid image path")
