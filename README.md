# 🍎 YOLO Fruit Detection App

Ứng dụng nhận dạng trái cây (apple, banana, orange) sử dụng YOLOv8 với giao diện PyQt5.

## ✨ Tính năng

- 🖼️ Nhận dạng từ ảnh tĩnh
- 📹 Camera realtime detection
- 🎨 Giao diện PyQt5 hiện đại
- 📊 Hiển thị confidence score và bounding box
- ⚙️ Điều chỉnh confidence threshold

## 📁 Cấu trúc

```
DA_DIP/
├── 1_dataset/          # Dataset và data.yaml
├── 2_training/         # Training scripts và models
├── 3_application/      # Ứng dụng chính
│   ├── app.py         # GUI PyQt5
│   ├── predict_image.py
│   ├── camera.py
│   ├── utils.py
│   └── model/best.pt
└── requirements.txt
```

## 🛠️ Cài đặt

```bash
git clone https://github.com/Htam0404/DA_DIP.git
cd DA_DIP
py -m pip install -r requirements.txt  # Windows
```

## 🚀 Sử dụng

**Giao diện PyQt5:**
```bash
cd 3_application
py app.py
```

**Command line:**
```bash
py predict_image.py  # Nhận dạng ảnh
py camera.py        # Camera realtime
```

**Phím tắt camera:** `q` (thoát) | `s` (lưu frame) | `+/-` (điều chỉnh threshold)

## 📦 Dependencies

Python 3.8+ • PyTorch • YOLOv8 • OpenCV • PyQt5 • NumPy

## 📊 Dataset

3 classes: 🍎 Apple | 🍌 Banana | 🍊 Orange

## 👥 Team

DA_DIP - Fruit Detection with YOLO • License: CC BY 4.0

