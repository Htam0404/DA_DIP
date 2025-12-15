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
- **Python 3.11.9** ([Download](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe)) 
- Visual C++ Redistributable ([Download](https://aka.ms/vs/17/release/vc_redist.x64.exe))

### Cài đặt nhanh

```bash
# 1. Clone repository
git clone https://github.com/Htam0404/DA_DIP.git
cd DA_DIP

# 2. Tạo virtual environment với Python 3.11
python -m venv venv
# Hoặc chỉ định đường dẫn cụ thể:
# C:\Users\<YourName>\AppData\Local\Programs\Python\Python311\python.exe -m venv venv

# 3. Kích hoạt virtual environment (Windows)
venv\Scripts\activate


# 4. Cài đặt dependencies với PyTorch 2.5.1
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m pip uninstall torch torchvision -y
venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu

# 5. Copy model vào app
copy 2_training\best.pt 3_application\model\best.pt  # Windows
```

### ⚠️ Xử lý lỗi DLL (Windows)

Nếu gặp lỗi `OSError: [WinError 1114] A dynamic link library (DLL) initialization failed`:

**Giải pháp đã test:**
1. **Cài Visual C++ Redistributable** ([Download](https://aka.ms/vs/17/release/vc_redist.x64.exe))
2. **Dùng Python 3.11.9** thay vì Python 3.10 hoặc 3.12
3. **Cài PyTorch 2.5.1** (phiên bản ổn định hơn 2.9.1):
   ```bash
   venv\Scripts\python.exe -m pip uninstall torch torchvision -y
   venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
   ```
4. Khởi động lại máy sau khi cài Visual C++

## 🚀 Sử dụng

**Giao diện PyQt5:**
```bash
cd 3_application
..\venv\Scripts\python.exe app.py
# Hoặc nếu đã activate venv:
# python app.py
```

**Command line:**
```bash
cd 3_application
..\venv\Scripts\python.exe predict_image.py  
..\venv\Scripts\python.exe camera.py        
```


**Phím tắt camera:** `q` (thoát) | `s` (lưu frame) | `+/-` (điều chỉnh threshold)

## 📦 Dependencies

- **Python 3.11.9** (khuyến nghị)
- **PyTorch 2.5.1** (CPU only - tương thích tốt)
- Ultralytics YOLO
- OpenCV
- PyQt5
- NumPy
- Visual C++ Redistributable

## 📊 Dataset

3 classes: 🍎 Apple | 🍌 Banana | 🍊 Orange


