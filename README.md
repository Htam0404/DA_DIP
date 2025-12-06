# Fruit-Classification-YOLO

Dự án nhận dạng và phân loại trái cây sử dụng YOLO (object detection) và mô-đun ứng dụng để demo.

Structure: hãy xem cây thư mục trong repo.

Nội dung chính:

- `dataset/` - ảnh và label cho train/valid/test
- `yolov11_training/` - cấu hình training, weights, kết quả
- `src/` - mã nguồn chính (load model, inference, xử lý ảnh)
- `app/` - giao diện người dùng
- `test/`, `report/`, `logs/` - tài liệu và nhật ký

Hướng dẫn nhanh:

1. Tạo virtualenv và cài dependencies: `pip install -r requirements.txt`
2. Chuẩn bị dataset theo cấu trúc `dataset/`
3. Chạy training (nếu có): `python yolov11_training/train.py`
4. Chạy ứng dụng demo: `python src/main.py`

Ghi chú: Các file trong thư mục là khung (skeleton) để bắt đầu. Thêm dữ liệu và weights để chạy thực tế.
