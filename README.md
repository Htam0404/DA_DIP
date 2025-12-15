# 🍎 YOLO Fruit Detection App

Ứng dụng nhận dạng trái cây (apple, banana, orange) sử dụng YOLOv8 với giao diện PyQt5.

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

### Yêu cầu hệ thống
- Python 3.8+ ([Download](https://www.python.org/downloads/))

### Cài đặt nhanh

```bash
# 1. Clone repository
git clone https://github.com/Htam0404/DA_DIP.git
cd DA_DIP

# 2. Tạo virtual environment (khuyến nghị)
python -m venv venv
venv\Scripts\activate  

# 3. Cài PyTorch CPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Cài các thư viện còn lại
pip install -r requirements.txt

# 5. Copy model vào app
copy 2_training\best.pt 3_application\model\best.pt  

```

### ⚠️ Xử lý lỗi DLL (Windows)

Nếu gặp lỗi `DLL initialization failed`:
1. Tải [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
2. Cài đặt và khởi động lại máy
3. Chạy lại `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

## 🚀 Sử dụng

**Giao diện PyQt5:**
```bash
cd 3_application
py app.py
```

**Command line:**
```bash
py predict_image.py  
py camera.py        
```

**Phím tắt camera:** `q` (thoát) | `s` (lưu frame) | `+/-` (điều chỉnh threshold)

## 📦 Dependencies

- Python 3.8+
- PyTorch (CPU only)
- Ultralytics YOLO
- OpenCV
- PyQt5
- NumPy

