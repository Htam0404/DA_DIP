import cv2
import numpy as np
from ultralytics import YOLO
import os
from utils import (
    draw_bounding_boxes,
    convert_yolo_results_to_boxes,
    generate_colors
)


class CameraPredictor:
    """Class để nhận dạng đối tượng từ camera realtime sử dụng YOLO"""
    
    def __init__(self, model_path: str, class_names: list):
        """
        Khởi tạo CameraPredictor
        
        Args:
            model_path: Đường dẫn đến file model YOLO (.pt)
            class_names: List tên các class
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        self.model.to('cpu')
        self.class_names = class_names
        self.colors = generate_colors(len(class_names))
        self.is_running = False
        print("Model loaded successfully (CPU mode)!")
        
    def run(self, camera_id: int = 0, confidence_threshold: float = 0.5, 
            display_fps: bool = True):
        """
        Chạy nhận dạng realtime từ camera
        
        Args:
            camera_id: ID của camera (0 cho camera mặc định)
            confidence_threshold: Ngưỡng confidence (0-1)
            display_fps: Có hiển thị FPS không
        """
        # Mở camera
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise Exception(f"Cannot open camera {camera_id}")
        
        print(f"\nCamera opened successfully!")
        print(f"Press 'q' to quit")
        print(f"Press 's' to save current frame")
        print(f"Press '+' to increase confidence threshold")
        print(f"Press '-' to decrease confidence threshold")
        
        self.is_running = True
        frame_count = 0
        
        # Để tính FPS
        import time
        prev_time = time.time()
        fps = 0
        
        while self.is_running:
            # Đọc frame
            ret, frame = cap.read()
            
            if not ret:
                print("Cannot read frame from camera")
                break
            
            # Predict
            results = self.model.predict(
                frame,
                conf=confidence_threshold,
                verbose=False
            )
            
            # Chuyển đổi results sang boxes
            boxes = convert_yolo_results_to_boxes(results)
            
            # Vẽ bounding boxes
            annotated_frame = draw_bounding_boxes(
                frame,
                boxes,
                self.class_names,
                self.colors,
                confidence_threshold
            )
            
            # Tính FPS
            if display_fps:
                current_time = time.time()
                fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
                prev_time = current_time
                
                # Vẽ FPS
                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            
            # Hiển thị confidence threshold
            cv2.putText(
                annotated_frame,
                f"Conf: {confidence_threshold:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Hiển thị số objects detected
            cv2.putText(
                annotated_frame,
                f"Objects: {len(boxes)}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Hiển thị frame
            cv2.imshow("YOLO Camera Detection", annotated_frame)
            
            # Xử lý phím
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Quit
                print("\nStopping camera...")
                break
            elif key == ord('s'):
                # Save frame
                filename = f"captured_frame_{frame_count}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"Frame saved as {filename}")
                frame_count += 1
            elif key == ord('+') or key == ord('='):
                # Tăng confidence threshold
                confidence_threshold = min(confidence_threshold + 0.05, 1.0)
                print(f"Confidence threshold: {confidence_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                # Giảm confidence threshold
                confidence_threshold = max(confidence_threshold - 0.05, 0.0)
                print(f"Confidence threshold: {confidence_threshold:.2f}")
        
        # Giải phóng resources
        self.is_running = False
        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed.")
    
    def stop(self):
        """Dừng camera"""
        self.is_running = False


class CameraThread:
    """Class để chạy camera trong thread riêng (dùng cho PyQt5)"""
    
    def __init__(self, model_path: str, class_names: list):
        """
        Khởi tạo CameraThread
        
        Args:
            model_path: Đường dẫn đến file model YOLO (.pt)
            class_names: List tên các class
        """
        self.predictor = CameraPredictor(model_path, class_names)
        self.camera_id = 0
        self.confidence_threshold = 0.5
        
    def set_camera_id(self, camera_id: int):
        """Set camera ID"""
        self.camera_id = camera_id
        
    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold"""
        self.confidence_threshold = threshold
        
    def start(self):
        """Bắt đầu camera"""
        self.predictor.run(
            camera_id=self.camera_id,
            confidence_threshold=self.confidence_threshold
        )
    
    def stop(self):
        """Dừng camera"""
        self.predictor.stop()


def main():
    """Hàm main để test"""
    # Đường dẫn model
    model_path = "model/best.pt"
    
    # Tên các class (lấy từ data.yaml)
    class_names = ['apple', 'banana', 'orange']
    
    # Khởi tạo camera predictor
    try:
        camera = CameraPredictor(model_path, class_names)
        
        # Chạy camera
        camera.run(camera_id=0, confidence_threshold=0.5)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
