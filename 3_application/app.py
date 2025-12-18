import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSlider, QGroupBox, QMessageBox,
    QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont
from predict_image import ImagePredictor
from camera import CameraPredictor
from utils import (
    convert_yolo_results_to_boxes, draw_bounding_boxes, 
    generate_colors, auto_threshold, calculate_optimal_threshold,
    non_max_suppression_custom
)


class CameraWorker(QThread):
    """Thread worker để xử lý camera realtime"""
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, model_path, class_names):
        super().__init__()
        self.model_path = model_path
        self.class_names = class_names
        self.colors = generate_colors(len(class_names))
        self.is_running = False
        self.camera_id = 0
        self.confidence_threshold = 0.5
        self.predictor = None
        
    def run(self):
        """Chạy camera detection"""
        from ultralytics import YOLO
        
        # Load model (CPU mode)
        self.predictor = YOLO(self.model_path)
        self.predictor.to('cpu')
        
        # Mở camera
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            return
        
        self.is_running = True
        
        while self.is_running:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Predict
            results = self.predictor.predict(
                frame,
                conf=self.confidence_threshold,
                verbose=False
            )
            
            # Vẽ bounding boxes
            boxes = convert_yolo_results_to_boxes(results)
            annotated_frame = draw_bounding_boxes(
                frame,
                boxes,
                self.class_names,
                self.colors,
                self.confidence_threshold
            )
            
            # Emit frame
            self.frame_ready.emit(annotated_frame)
        
        cap.release()
    
    def stop(self):
        """Dừng camera"""
        self.is_running = False
        self.wait()


class YOLOApp(QMainWindow):
    """Giao diện chính của ứng dụng YOLO"""
    
    def __init__(self):
        super().__init__()
        
        # Thiết lập các biến
        self.model_path = "model/best.pt"
        self.class_names = ['apple', 'banana', 'orange']
        self.confidence_threshold = 0.70  # Tăng threshold mặc định
        self.current_image = None
        self.camera_worker = None
        self.auto_threshold_enabled = False  
        
        # Khởi tạo predictor
        try:
            self.image_predictor = ImagePredictor(self.model_path, self.class_names)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot load model: {e}")
            sys.exit(1)
        
        # Thiết lập UI
        self.init_ui()
        
    def init_ui(self):
        """Khởi tạo giao diện"""
        self.setWindowTitle("YOLO Object Detection - Fruit Detector")
        self.setGeometry(100, 100, 1200, 800)
        
        # Widget chính
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout chính
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Panel bên trái - Controls
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel)
        
        # Panel bên phải - Display
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel)
        
        # Set ratio
        main_layout.setStretch(0, 1)  # Left panel
        main_layout.setStretch(1, 3)  # Right panel
        
    def create_left_panel(self):
        """Tạo panel điều khiển bên trái"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Title
        title = QLabel("YOLO Detector")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Model info
        model_group = QGroupBox("Model Information")
        model_layout = QVBoxLayout()
        model_info = QLabel(f"Model: {os.path.basename(self.model_path)}\n"
                           f"Classes: {', '.join(self.class_names)}")
        model_layout.addWidget(model_info)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Confidence slider
        slider_group = QGroupBox("Confidence Threshold")
        slider_layout = QVBoxLayout()
        
        self.conf_label = QLabel(f"Threshold: {self.confidence_threshold:.2f}")
        slider_layout.addWidget(self.conf_label)
        
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(0)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(70)  # Mặc định 0.70
        self.conf_slider.valueChanged.connect(self.on_confidence_changed)
        slider_layout.addWidget(self.conf_slider)
        
        # Auto threshold checkbox
        self.auto_threshold_checkbox = QCheckBox("🤖 Auto Threshold")
        self.auto_threshold_checkbox.setToolTip("Tự động điều chỉnh threshold dựa trên detections")
        self.auto_threshold_checkbox.stateChanged.connect(self.on_auto_threshold_toggled)
        slider_layout.addWidget(self.auto_threshold_checkbox)
        
        slider_group.setLayout(slider_layout)
        layout.addWidget(slider_group)
        
        # Image Detection Group
        image_group = QGroupBox("Image Detection")
        image_layout = QVBoxLayout()
        
        self.btn_choose_image = QPushButton("📁 Choose Image")
        self.btn_choose_image.setMinimumHeight(50)
        self.btn_choose_image.clicked.connect(self.choose_image)
        image_layout.addWidget(self.btn_choose_image)
        
        self.btn_detect_image = QPushButton("🔍 Detect Objects")
        self.btn_detect_image.setMinimumHeight(50)
        self.btn_detect_image.setEnabled(False)
        self.btn_detect_image.clicked.connect(self.detect_image)
        image_layout.addWidget(self.btn_detect_image)
        
        image_group.setLayout(image_layout)
        layout.addWidget(image_group)
        
        # Camera Detection Group
        camera_group = QGroupBox("Camera Detection")
        camera_layout = QVBoxLayout()
        
        self.btn_start_camera = QPushButton("📹 Start Camera")
        self.btn_start_camera.setMinimumHeight(50)
        self.btn_start_camera.clicked.connect(self.start_camera)
        camera_layout.addWidget(self.btn_start_camera)
        
        self.btn_stop_camera = QPushButton("⏹ Stop Camera")
        self.btn_stop_camera.setMinimumHeight(50)
        self.btn_stop_camera.setEnabled(False)
        self.btn_stop_camera.clicked.connect(self.stop_camera)
        camera_layout.addWidget(self.btn_stop_camera)
        
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        # Detection Info
        self.info_group = QGroupBox("Detection Info")
        self.info_layout = QVBoxLayout()
        self.info_label = QLabel("No detection yet")
        self.info_label.setWordWrap(True)
        self.info_layout.addWidget(self.info_label)
        self.info_group.setLayout(self.info_layout)
        layout.addWidget(self.info_group)
        
        # Spacer
        layout.addStretch()
        
        # Exit button
        btn_exit = QPushButton("❌ Exit")
        btn_exit.setMinimumHeight(40)
        btn_exit.clicked.connect(self.close)
        layout.addWidget(btn_exit)
        
        return panel
    
    def create_right_panel(self):
        """Tạo panel hiển thị bên phải"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Display label
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("QLabel { background-color: #2b2b2b; border: 2px solid #555; }")
        self.display_label.setMinimumSize(800, 600)
        self.display_label.setText("No image loaded\n\nChoose an image or start camera")
        self.display_label.setStyleSheet("QLabel { background-color: #2b2b2b; color: white; font-size: 18px; }")
        
        layout.addWidget(self.display_label)
        
        return panel
    
    def on_confidence_changed(self, value):
        """Xử lý khi thay đổi confidence threshold"""
        if not self.auto_threshold_enabled:  # Chỉ update khi không dùng auto
            self.confidence_threshold = value / 100.0
            self.conf_label.setText(f"Threshold: {self.confidence_threshold:.2f}")
        
        # Cập nhật cho camera worker nếu đang chạy
        if self.camera_worker and self.camera_worker.isRunning():
            self.camera_worker.confidence_threshold = self.confidence_threshold
        
        # Re-detect nếu đang có ảnh
        if self.current_image is not None and not self.auto_threshold_enabled:
            self.detect_image()
    
    def on_auto_threshold_toggled(self, state):
        """Xử lý khi bật/tắt auto threshold"""
        self.auto_threshold_enabled = (state == Qt.Checked)
        
        if self.auto_threshold_enabled:
            # Disable slider khi dùng auto
            self.conf_slider.setEnabled(False)
            self.conf_label.setText("Threshold: AUTO")
            
            # Re-detect với auto threshold nếu đang có ảnh
            if self.current_image is not None:
                self.detect_image()
        else:
            # Enable slider khi tắt auto
            self.conf_slider.setEnabled(True)
            self.confidence_threshold = self.conf_slider.value() / 100.0
            self.conf_label.setText(f"Threshold: {self.confidence_threshold:.2f}")
            
            # Re-detect với threshold manual
            if self.current_image is not None:
                self.detect_image()
    
    def choose_image(self):
        """Chọn ảnh từ máy"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            # Load và hiển thị ảnh gốc
            self.current_image = cv2.imread(file_path)
            
            if self.current_image is not None:
                self.display_image(self.current_image)
                self.btn_detect_image.setEnabled(True)
                self.info_label.setText(f"Image loaded: {os.path.basename(file_path)}")
            else:
                QMessageBox.warning(self, "Error", "Cannot load image!")
    
    def detect_image(self):
        """Nhận dạng đối tượng trong ảnh"""
        if self.current_image is None:
            return
        
        try:
            # Predict với threshold người dùng chọn
            results = self.image_predictor.model.predict(
                self.current_image,
                conf=self.confidence_threshold,
                verbose=False
            )
            
            # Chuyển đổi results sang boxes
            boxes = convert_yolo_results_to_boxes(results)
            
            # Tính auto threshold nếu được bật
            if self.auto_threshold_enabled and boxes:
                self.confidence_threshold = calculate_optimal_threshold(boxes, method='adaptive')
                self.conf_label.setText(f"Threshold: AUTO ({self.confidence_threshold:.2f})")
                
                # Update slider (nhưng không trigger event)
                self.conf_slider.blockSignals(True)
                self.conf_slider.setValue(int(self.confidence_threshold * 100))
                self.conf_slider.blockSignals(False)
            
            # Vẽ bounding boxes với threshold đã tính
            annotated_image = draw_bounding_boxes(
                self.current_image,
                boxes,
                self.class_names,
                self.image_predictor.colors,
                self.confidence_threshold
            )
            
            # Hiển thị
            self.display_image(annotated_image)
            
            # Đếm số objects sau khi lọc threshold
            filtered_boxes = [b for b in boxes if b[4] >= self.confidence_threshold]
            
            # Cập nhật info
            if len(filtered_boxes) > 0:
                info_text = f"Detected {len(filtered_boxes)} objects:\n\n"
                for i, box in enumerate(filtered_boxes, 1):
                    x1, y1, x2, y2, conf, cls = box
                    class_name = self.class_names[int(cls)]
                    info_text += f"{i}. {class_name}: {conf:.2f}\n"
                self.info_label.setText(info_text)
            else:
                self.info_label.setText("No objects detected")
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Detection error: {e}")
    
    def start_camera(self):
        """Bắt đầu camera detection"""
        # Tắt các nút image
        self.btn_choose_image.setEnabled(False)
        self.btn_detect_image.setEnabled(False)
        self.btn_start_camera.setEnabled(False)
        self.btn_stop_camera.setEnabled(True)
        
        # Tạo và chạy camera worker
        self.camera_worker = CameraWorker(self.model_path, self.class_names)
        self.camera_worker.confidence_threshold = self.confidence_threshold
        self.camera_worker.frame_ready.connect(self.display_image)
        self.camera_worker.start()
        
        self.info_label.setText("Camera is running...")
    
    def stop_camera(self):
        """Dừng camera detection"""
        if self.camera_worker:
            self.camera_worker.stop()
            self.camera_worker = None
        
        # Bật lại các nút
        self.btn_choose_image.setEnabled(True)
        if self.current_image is not None:
            self.btn_detect_image.setEnabled(True)
        self.btn_start_camera.setEnabled(True)
        self.btn_stop_camera.setEnabled(False)
        
        self.info_label.setText("Camera stopped")
        self.display_label.setText("Camera stopped\n\nChoose an image or start camera")
    
    def display_image(self, image):
        """Hiển thị ảnh lên label"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get display size
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        # Convert to QImage
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.display_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.display_label.setPixmap(scaled_pixmap)
    
    def closeEvent(self, event):
        """Xử lý khi đóng ứng dụng"""
        if self.camera_worker:
            self.camera_worker.stop()
        event.accept()


def main():
    """Hàm main"""
    app = QApplication(sys.argv)
    
    # Set dark theme
    app.setStyle("Fusion")
    
    window = YOLOApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
