import cv2
import numpy as np
from typing import List, Tuple
import random


def generate_colors(num_classes: int) -> List[Tuple[int, int, int]]:
    """
    Tạo màu sắc ngẫu nhiên cho mỗi class
    
    Args:
        num_classes: Số lượng classes
        
    Returns:
        List màu RGB cho mỗi class
    """
    # Màu sắc nhẹ nhàng, dễ nhìn
    predefined_colors = [
        (34, 139, 34),     # Xanh lá forest green - Apple
        (255, 165, 0),     # Cam nhạt - Banana  
        (255, 99, 71),     # Đỏ tomato - Orange
        (70, 130, 180),    # Xanh steel blue
        (220, 20, 60),     # Đỏ crimson
        (147, 112, 219),   # Tím medium purple
    ]
    
    colors = []
    for i in range(num_classes):
        if i < len(predefined_colors):
            colors.append(predefined_colors[i])
        else:
            # Nếu có nhiều hơn 6 class, tạo màu ngẫu nhiên
            random.seed(42 + i)
            colors.append((
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(100, 255)
            ))
    return colors


def non_max_suppression_custom(boxes: List[Tuple], iou_threshold: float = 0.5) -> List[Tuple]:
    """
    Non-Maximum Suppression để loại bỏ các boxes trùng lặp
    
    Args:
        boxes: List các box (x1, y1, x2, y2, confidence, class_id)
        iou_threshold: Ngưỡng IoU để coi là trùng lặp
        
    Returns:
        List các boxes sau khi loại bỏ trùng lặp
    """
    if not boxes:
        return []
    
    boxes_array = np.array(boxes)
    
 
    x1 = boxes_array[:, 0]
    y1 = boxes_array[:, 1]
    x2 = boxes_array[:, 2]
    y2 = boxes_array[:, 3]
    scores = boxes_array[:, 4]
    classes = boxes_array[:, 5]
    
    areas = (x2 - x1) * (y2 - y1)
    
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        
        same_class = classes[i] == classes[order[1:]]
        inds = np.where((iou <= iou_threshold) | (~same_class))[0]
        order = order[inds + 1]
    
    return [boxes[i] for i in keep]


def draw_bounding_boxes(image: np.ndarray, 
                       boxes: List[Tuple], 
                       class_names: List[str],
                       colors: List[Tuple[int, int, int]],
                       confidence_threshold: float = 0.5,
                       apply_nms: bool = True,
                       max_detections: int = 50) -> np.ndarray:
    """
    Vẽ bounding boxes và labels lên ảnh
    
    Args:
        image: Ảnh đầu vào (numpy array)
        boxes: List các box, mỗi box là tuple (x1, y1, x2, y2, confidence, class_id)
        class_names: List tên các class
        colors: List màu sắc cho mỗi class
        confidence_threshold: Ngưỡng confidence để hiển thị
        apply_nms: Có áp dụng NMS để loại bỏ boxes trùng lặp không
        max_detections: Số lượng detections tối đa hiển thị
        
    Returns:
        Ảnh đã vẽ bounding boxes
    """
    img_copy = image.copy()
    
    # Lọc theo confidence threshold
    filtered_boxes = [box for box in boxes if box[4] >= confidence_threshold]
    
    # Áp dụng NMS để loại bỏ boxes trùng lặp (IoU cao = loại bỏ nhiều)
    if apply_nms and filtered_boxes:
        filtered_boxes = non_max_suppression_custom(filtered_boxes, iou_threshold=0.3)
    
    # Sắp xếp theo confidence giảm dần và chỉ lấy top detections
    filtered_boxes = sorted(filtered_boxes, key=lambda x: x[4], reverse=True)[:max_detections]
    
    for box in filtered_boxes:
        x1, y1, x2, y2, conf, cls = box
        
        # Chuyển đổi sang int
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls = int(cls)
        
        # Lấy màu và tên class
        color = colors[cls] if cls < len(colors) else (0, 255, 0)
        label = class_names[cls] if cls < len(class_names) else f"Class {cls}"
        
        # Vẽ bounding box mảnh
        thickness = 2
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
        
        # Chỉ hiển thị tên class, không hiển thị confidence
        text = f"{label}"
        
        # Font nhỏ hơn
        font_scale = 0.5
        font_thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        
        # Padding cho label rất nhỏ
        padding = 4
        label_height = text_height + baseline + padding * 2
        label_width = text_width + padding * 2
        
        # Đảm bảo label không vượt ra ngoài ảnh
        label_y1 = max(0, y1 - label_height)
        label_y2 = label_y1 + label_height
        
        # Vẽ background cho text với opacity (semi-transparent)
        overlay = img_copy.copy()
        cv2.rectangle(
            overlay,
            (x1, label_y1),
            (x1 + label_width, label_y2),
            color,
            -1  # Filled
        )
        # Blend với opacity 0.7
        cv2.addWeighted(overlay, 0.7, img_copy, 0.3, 0, img_copy)
        
        # Vẽ text với màu trắng và shadow để dễ đọc
        text_y = label_y1 + text_height + padding
        
        # Shadow (màu đen)
        cv2.putText(
            img_copy,
            text,
            (x1 + padding + 1, text_y + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),  # Đen
            font_thickness,
            cv2.LINE_AA
        )
        
        # Text chính (màu trắng)
        cv2.putText(
            img_copy,
            text,
            (x1 + padding, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),  # Trắng
            font_thickness,
            cv2.LINE_AA
        )
    
    return img_copy
    
    return img_copy


def resize_image_keep_aspect(image: np.ndarray, max_width: int = 800, max_height: int = 600) -> np.ndarray:
    """
    Resize ảnh giữ nguyên tỷ lệ
    
    Args:
        image: Ảnh đầu vào
        max_width: Chiều rộng tối đa
        max_height: Chiều cao tối đa
        
    Returns:
        Ảnh đã resize
    """
    h, w = image.shape[:2]
    
    # Tính tỷ lệ scale
    scale = min(max_width / w, max_height / h, 1.0)
    
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return image


def convert_yolo_results_to_boxes(results) -> List[Tuple]:
    """
    Chuyển đổi kết quả từ YOLO sang format boxes
    
    Args:
        results: Kết quả từ YOLO model
        
    Returns:
        List các box (x1, y1, x2, y2, confidence, class_id)
    """
    boxes = []
    
    if results and len(results) > 0:
        result = results[0]
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                # Lấy tọa độ xyxy
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # Lấy confidence
                conf = float(box.conf[0].cpu().numpy())
                # Lấy class id
                cls = int(box.cls[0].cpu().numpy())
                
                boxes.append((x1, y1, x2, y2, conf, cls))
    
    return boxes


def auto_threshold(boxes: List[Tuple], target_detections: int = 3) -> float:
    """
    Tự động tính threshold tối ưu dựa trên số detections mong muốn
    
    Args:
        boxes: List các box (x1, y1, x2, y2, confidence, class_id)
        target_detections: Số lượng detections mong muốn (mặc định: 3)
        
    Returns:
        Threshold tối ưu (0.0 - 1.0)
    """
    if not boxes:
        return 0.5  # Default nếu không có detection
    
    # Sắp xếp theo confidence giảm dần
    sorted_boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    
    # Nếu có ít detection hơn target, dùng threshold thấp
    if len(sorted_boxes) <= target_detections:
        return 0.3
    
    # Lấy confidence của detection thứ target_detections
    threshold = sorted_boxes[target_detections - 1][4]
    
    # Đảm bảo threshold trong khoảng hợp lý (0.3 - 0.8)
    threshold = max(0.3, min(0.8, threshold))
    
    return round(threshold, 2)


def calculate_optimal_threshold(boxes: List[Tuple], method: str = 'mean') -> float:
    """
    Tính threshold tối ưu dựa trên confidence scores
    
    Args:
        boxes: List các box
        method: Phương pháp tính ('mean', 'median', 'adaptive')
        
    Returns:
        Threshold tối ưu
    """
    if not boxes:
        return 0.5
    
    confidences = [box[4] for box in boxes]
    
    if method == 'mean':
        # Trung bình confidence
        threshold = np.mean(confidences)
    elif method == 'median':
        # Trung vị confidence
        threshold = np.median(confidences)
    elif method == 'adaptive':
        # Adaptive: mean - std (loại bỏ outliers thấp)
        mean_conf = np.mean(confidences)
        std_conf = np.std(confidences)
        threshold = max(0.3, mean_conf - std_conf)
    else:
        threshold = 0.5
    
    # Đảm bảo trong khoảng 0.2 - 0.9
    threshold = max(0.2, min(0.9, threshold))
    
    return round(threshold, 2)
