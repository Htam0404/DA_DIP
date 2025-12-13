import cv2
import numpy as np
from ultralytics import YOLO
import os
from utils import (
    draw_bounding_boxes,
    convert_yolo_results_to_boxes,
    generate_colors,
    resize_image_keep_aspect
)


class ImagePredictor:
    """Class để nhận dạng đối tượng trong ảnh sử dụng YOLO"""
    
    def __init__(self, model_path: str, class_names: list):
        """
        Khởi tạo ImagePredictor
        
        Args:
            model_path: Đường dẫn đến file model YOLO (.pt)
            class_names: List tên các class
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        self.class_names = class_names
        self.colors = generate_colors(len(class_names))
        print("Model loaded successfully!")
        
    def predict(self, image_path: str, confidence_threshold: float = 0.5, 
                save_result: bool = False, output_path: str = None) -> tuple:
        """
        Nhận dạng đối tượng trong ảnh
        
        Args:
            image_path: Đường dẫn đến ảnh
            confidence_threshold: Ngưỡng confidence (0-1)
            save_result: Có lưu kết quả không
            output_path: Đường dẫn lưu kết quả (nếu None sẽ tự động tạo)
            
        Returns:
            tuple: (annotated_image, boxes, results)
        """
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Predict
        results = self.model.predict(
            image,
            conf=confidence_threshold,
            verbose=False
        )
        
        # Chuyển đổi results sang boxes
        boxes = convert_yolo_results_to_boxes(results)
        
        # Vẽ bounding boxes
        annotated_image = draw_bounding_boxes(
            image,
            boxes,
            self.class_names,
            self.colors,
            confidence_threshold
        )
        
        # Lưu kết quả nếu cần
        if save_result:
            if output_path is None:
                # Tự động tạo tên file output
                base_name = os.path.basename(image_path)
                name, ext = os.path.splitext(base_name)
                output_path = f"result_{name}{ext}"
            
            cv2.imwrite(output_path, annotated_image)
            print(f"Result saved to {output_path}")
        
        return annotated_image, boxes, results
    
    def predict_and_display(self, image_path: str, confidence_threshold: float = 0.5):
        """
        Nhận dạng và hiển thị kết quả
        
        Args:
            image_path: Đường dẫn đến ảnh
            confidence_threshold: Ngưỡng confidence
        """
        annotated_image, boxes, _ = self.predict(image_path, confidence_threshold)
        
        # Resize để hiển thị
        display_image = resize_image_keep_aspect(annotated_image, 1200, 800)
        
        # Hiển thị
        cv2.imshow(f"Detection Results - {os.path.basename(image_path)}", display_image)
        print(f"\nDetected {len(boxes)} objects:")
        for i, box in enumerate(boxes, 1):
            x1, y1, x2, y2, conf, cls = box
            class_name = self.class_names[int(cls)]
            print(f"  {i}. {class_name}: {conf:.2f}")
        
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Hàm main để test"""
    # Đường dẫn model
    model_path = "model/best.pt"
    
    # Tên các class (lấy từ data.yaml)
    class_names = ['apple', 'banana', 'orange']
    
    # Khởi tạo predictor
    try:
        predictor = ImagePredictor(model_path, class_names)
        
        # Test với ảnh mẫu (bạn cần có ảnh để test)
        # predictor.predict_and_display("test_image.jpg", confidence_threshold=0.5)
        
        print("\nImagePredictor initialized successfully!")
        print("Use predictor.predict(image_path) to detect objects in an image")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
