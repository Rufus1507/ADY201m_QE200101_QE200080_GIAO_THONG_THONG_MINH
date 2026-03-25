import sqlite3
import json
import os
import sys
from datetime import datetime, timezone
import pandas as pd
from minio import Minio
from minio.error import S3Error

# ================= CONFIG =================
SQLITE_DB = "data/raw/data_traffic_QN.db"
CHECKPOINT_FILE = "checkpoint/traffic_checkpoint.json"
EXPORT_DIR = "export"
CLEANUP_AFTER_UPLOAD = True  # Xóa file tạm sau khi upload thành công

BUCKET_NAME = "raw-traffic-data"
OBJECT_PREFIX = "traffic/incremental"
LOCATIONS_PREFIX = "traffic/locations"

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "admin123")


def load_checkpoint():
    """Load checkpoint từ file JSON"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
                return checkpoint.get("last_uploaded_ts", "1970-01-01T00:00:00")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Lỗi đọc checkpoint: {e}. Sử dụng giá trị mặc định.")
            return "1970-01-01T00:00:00"
    return "1970-01-01T00:00:00"


def save_checkpoint(timestamp):
    """Lưu checkpoint mới"""
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_uploaded_ts": timestamp}, f, indent=2)


def read_new_data(last_ts):
    """Đọc dữ liệu mới từ SQLite"""
    if not os.path.exists(SQLITE_DB):
        print(f"❌ Không tìm thấy database: {SQLITE_DB}")
        return None
    
    try:
        conn = sqlite3.connect(SQLITE_DB)
        query = """
            SELECT *
            FROM traffic_data
            WHERE timestamp > ?
            ORDER BY timestamp
        """
        df = pd.read_sql_query(query, conn, params=(last_ts,))
        conn.close()
        return df
    except sqlite3.Error as e:
        print(f"❌ Lỗi đọc database: {e}")
        return None


def read_locations():
    """Đọc toàn bộ dữ liệu bảng locations từ SQLite"""
    if not os.path.exists(SQLITE_DB):
        print(f"❌ Không tìm thấy database: {SQLITE_DB}")
        return None

    try:
        conn = sqlite3.connect(SQLITE_DB)
        df = pd.read_sql_query("SELECT * FROM locations", conn)
        conn.close()
        return df
    except sqlite3.Error as e:
        print(f"❌ Lỗi đọc bảng locations: {e}")
        return None


def export_to_parquet(df, prefix="traffic"):
    """Xuất DataFrame ra file Parquet"""
    os.makedirs(EXPORT_DIR, exist_ok=True)
    now = datetime.now(timezone.utc)
    file_name = f"{prefix}_{now.strftime('%Y%m%d_%H%M%S')}.parquet"
    file_path = os.path.join(EXPORT_DIR, file_name)

    df.to_parquet(file_path, index=False)
    return file_path, file_name, now


def upload_locations_to_minio():
    """Đọc bảng locations và upload lên MinIO"""
    print("\n📍 Bắt đầu upload bảng locations...")

    df_loc = read_locations()
    if df_loc is None:
        print("⚠️ Bỏ qua upload locations do lỗi đọc dữ liệu.")
        return

    if df_loc.empty:
        print("⚠️ Bảng locations rỗng, bỏ qua.")
        return

    print(f"📊 Tìm thấy {len(df_loc)} bản ghi trong bảng locations")

    file_path, file_name, now = export_to_parquet(df_loc, prefix="locations")
    print(f"📄 Đã xuất file locations: {file_path}")

    try:
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )

        if not client.bucket_exists(BUCKET_NAME):
            client.make_bucket(BUCKET_NAME)
            print(f"📁 Đã tạo bucket: {BUCKET_NAME}")

        object_name = f"{LOCATIONS_PREFIX}/{file_name}"
        client.fput_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
            file_path=file_path,
            content_type="application/octet-stream"
        )
        print(f"✅ Đã upload locations: {object_name}")
    except S3Error as e:
        print(f"❌ Lỗi MinIO khi upload locations: {e}")
    except Exception as e:
        print(f"❌ Lỗi kết nối MinIO khi upload locations: {e}")
    finally:
        if CLEANUP_AFTER_UPLOAD:
            cleanup_file(file_path)


def upload_to_minio(file_path, file_name, timestamp):
    """Upload file lên MinIO"""
    try:
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )
        
        # Tạo bucket nếu chưa tồn tại
        if not client.bucket_exists(BUCKET_NAME):
            client.make_bucket(BUCKET_NAME)
            print(f"📁 Đã tạo bucket: {BUCKET_NAME}")
        
        # Tạo object name với partition theo ngày
        date_partition = timestamp.strftime("%Y-%m-%d")
        object_name = f"{OBJECT_PREFIX}/{date_partition}/{file_name}"
        
        # Upload file
        client.fput_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
            file_path=file_path,
            content_type="application/octet-stream"
        )
        
        return object_name
    
    except S3Error as e:
        print(f"❌ Lỗi MinIO: {e}")
        return None
    except Exception as e:
        print(f"❌ Lỗi kết nối MinIO: {e}")
        return None


def cleanup_file(file_path):
    """Xóa file tạm sau khi upload"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"🗑️ Đã xóa file tạm: {file_path}")
    except OSError as e:
        print(f"⚠️ Không thể xóa file tạm: {e}")


def main():
    """Hàm chính"""
    print("=" * 50)
    print("🚀 Bắt đầu upload dữ liệu lên MinIO")
    print("=" * 50)

    # ── 1. Upload bảng locations ──────────────────────────
    upload_locations_to_minio()

    # ── 2. Upload traffic_data tăng dần ──────────────────
    print("\n📡 Bắt đầu upload bảng traffic_data...")

    # Load checkpoint
    last_uploaded_ts = load_checkpoint()
    print(f"📌 Timestamp cuối cùng đã upload: {last_uploaded_ts}")

    # Đọc dữ liệu mới
    df = read_new_data(last_uploaded_ts)
    if df is None:
        sys.exit(1)

    if df.empty:
        print("✅ Không có dữ liệu traffic mới để upload")
        sys.exit(0)

    print(f"📊 Tìm thấy {len(df)} bản ghi mới")

    # Export ra file parquet
    file_path, file_name, now = export_to_parquet(df, prefix="traffic")
    print(f"📄 Đã xuất file: {file_path}")

    # Upload lên MinIO
    object_name = upload_to_minio(file_path, file_name, now)
    if object_name is None:
        print("❌ Upload thất bại!")
        sys.exit(1)

    # Cập nhật checkpoint
    new_last_ts = df["timestamp"].max()
    save_checkpoint(new_last_ts)

    # Dọn dẹp file tạm
    if CLEANUP_AFTER_UPLOAD:
        cleanup_file(file_path)

    print("=" * 50)
    print("✅ Upload hoàn tất!")
    print(f"📦 Traffic object: {object_name}")
    print(f"🕒 Checkpoint mới: {new_last_ts}")
    print("=" * 50)


if __name__ == "__main__":
    main()
