# 1. Chọn hệ điều hành nền: Tải một bản Linux siêu nhẹ đã cài sẵn Python 3.11
FROM python:3.11-slim

# 2. Tạo một thư mục tên là /app bên trong máy ảo và di chuyển vào đó
WORKDIR /app

# 3. Cài đặt các công cụ hệ thống cần thiết (để chạy SQLite và cài các thư viện Machine Learning)
RUN apt-get update && apt-get install -y \
    sqlite3 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy file danh sách thư viện từ máy bạn vào thư mục /app của máy ảo
COPY requirements.txt .

# 5. Ra lệnh cho máy ảo đọc file đó và tự động tải các thư viện (pandas, scikit-learn...) về
RUN pip install --no-cache-dir -r requirements.txt

# 6. Sau khi cài xong thư viện, copy toàn bộ thư mục code (src, utils...) từ máy bạn vào máy ảo
COPY . .

# 7. Cấu hình đường dẫn để Python hiểu được cấu trúc thư mục của bạn (tránh lỗi ModuleNotFoundError)
ENV PYTHONPATH=/app