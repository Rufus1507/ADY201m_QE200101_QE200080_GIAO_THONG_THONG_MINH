import sqlite3
from datetime import datetime
import os
import io
import pandas as pd
from minio import Minio

# ================= CONFIG =================
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "admin123"
BUCKET_NAME = "raw-traffic-data"
PREFIX = "traffic/incremental"
LOCATIONS_PREFIX = "traffic/locations"

CLEAN_DIR = "data/clean"
CLEAN_DB_PATH = os.path.join(CLEAN_DIR, "data_traffic_clean.db")

os.makedirs(CLEAN_DIR, exist_ok=True)

# ================= MINIO CLIENT =================
client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

# ================= SQLITE CLEAN TARGET =================
conn = sqlite3.connect(CLEAN_DB_PATH)
cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS traffic_data_clean (
        id INTEGER,
        timestamp TEXT,
        location TEXT,
        current_speed_kmh REAL,
        free_flow_speed_kmh REAL,
        speed_ratio REAL,
        traffic_level TEXT,
        confidence REAL,
        PRIMARY KEY (id, timestamp)
    )
""")

cur.execute("""
    CREATE TABLE IF NOT EXISTS locations (
        id INTEGER PRIMARY KEY,
        name TEXT,
        latitude REAL,
        longitude REAL
    )
""")

# ================= LOAD LOCATIONS FROM MINIO =================
print("📍 Loading locations table from MinIO...")
try:
    if client.bucket_exists(BUCKET_NAME):
        loc_objects = list(client.list_objects(BUCKET_NAME, prefix=LOCATIONS_PREFIX, recursive=True))
        loc_files = sorted(
            [obj.object_name for obj in loc_objects if obj.object_name.endswith('.parquet')]
        )
        if loc_files:
            latest_loc = loc_files[-1]  # lấy file mới nhất
            response = client.get_object(BUCKET_NAME, latest_loc)
            df_loc = pd.read_parquet(io.BytesIO(response.read()))
            response.close()
            df_loc.to_sql("locations", conn, if_exists="replace", index=False)
            print(f"✅ Đã load {len(df_loc)} bản ghi locations từ {latest_loc}")
        else:
            print("⚠️ Không tìm thấy file locations trên MinIO.")
    else:
        print(f"❌ Bucket {BUCKET_NAME} không tồn tại.")
except Exception as e:
    print(f"❌ Lỗi load locations: {e}")

# ================= FIND ALL PARQUET FILES =================
print("🔍 Searching for parquet files on MinIO...")
if client.bucket_exists(BUCKET_NAME):
    objects = client.list_objects(BUCKET_NAME, prefix=PREFIX, recursive=True)
    parquet_files = [obj.object_name for obj in objects if obj.object_name.endswith('.parquet')]
else:
    print(f"❌ Bucket {BUCKET_NAME} does not exist.")
    parquet_files = []

print(f"🔍 Found {len(parquet_files)} parquet files")

# ================= CLEAN LOOP =================
for obj_name in parquet_files:
    print(f"➡ Processing: {obj_name}")
    try:
        response = client.get_object(BUCKET_NAME, obj_name)
        file_data = response.read()
        df_raw = pd.read_parquet(io.BytesIO(file_data))
    except Exception as e:
        print(f"❌ Lỗi đọc file {obj_name}: {e}")
        continue
    finally:
        if 'response' in locals() and hasattr(response, 'close'):
            response.close()
        elif 'response' in locals() and hasattr(response, 'release_conn'):
            response.release_conn()

    for _, row in df_raw.iterrows():
        rid = row["id"]
        ts = row["timestamp"]
        location = row["location"]
        current_speed = row["current_speed_kmh"]
        free_flow_speed = row["free_flow_speed_kmh"]
        speed_ratio = row["speed_ratio"]
        traffic_level = row["traffic_level"]
        confidence = row.get("confidence", None)

        # ----- BASIC VALIDATION -----
        if pd.isna(current_speed) or pd.isna(free_flow_speed):
            continue
        if free_flow_speed <= 0:
            continue
        if pd.isna(speed_ratio) or pd.isna(traffic_level):
            continue
        if pd.isna(confidence) or confidence <= 0.9:
            continue
        # ----- NORMALIZE TIMESTAMP -----
        try:
            ts = datetime.fromisoformat(str(ts)).isoformat()
        except Exception:
            continue

        # ----- IDEMPOTENT INSERT -----
        cur.execute("""
            INSERT OR IGNORE INTO traffic_data_clean (
                id,
                timestamp,
                location,
                current_speed_kmh,
                free_flow_speed_kmh,
                speed_ratio,
                traffic_level,
                confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rid,
            ts,
            location,
            current_speed,
            free_flow_speed,
            speed_ratio,
            traffic_level,
            confidence
        ))

conn.commit()
conn.close()

print("✅ Incremental clean completed successfully")
print(f"📁 Clean DB: {CLEAN_DB_PATH}")