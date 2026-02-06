# Hướng dẫn chạy server OFFLINE (không có internet)

## Bước 1: Tải model về trước (Khi có internet)

Chạy script này **MỘT LẦN** khi bạn có internet để tải model về cache:

```bash
cd backend
python download_model.py
```

Script này sẽ:
- Tải model `google/translategemma-4b-it` (~8GB) về cache local
- Test load model để đảm bảo mọi thứ hoạt động
- Model sẽ được lưu tại: `~/.cache/huggingface/hub/`

**Lưu ý:**
- Cần ~10GB dung lượng ổ đĩa
- Quá trình tải có thể mất 30-60 phút tùy tốc độ mạng
- Đảm bảo có internet ổn định

## Bước 2: Chạy server offline

### Windows:
```bash
cd backend
run_offline.bat
```

### Linux/Mac:
```bash
cd backend
chmod +x run_offline.sh
./run_offline.sh
```

### Hoặc chạy thủ công:
```bash
# Windows PowerShell
$env:HF_HUB_OFFLINE="true"
$env:TRANSFORMERS_OFFLINE="true"
cd backend
.my-env\Scripts\activate
python main.py

# Linux/Mac
export HF_HUB_OFFLINE=true
export TRANSFORMERS_OFFLINE=true
cd backend
source .my-env/bin/activate
python main.py
```

## Kiểm tra model đã được cache chưa

Chạy lệnh này để kiểm tra:

```bash
python -c "from huggingface_hub import snapshot_download; import os; cache = os.path.expanduser('~/.cache/huggingface/hub'); print('Cache dir:', cache); print('Exists:', os.path.exists(cache))"
```

## Troubleshooting

### Lỗi: "Model not found in cache"
- **Nguyên nhân:** Model chưa được tải về
- **Giải pháp:** Chạy `python download_model.py` khi có internet

### Lỗi: "Connection refused" hoặc "Network error"
- **Nguyên nhân:** Server vẫn đang cố kết nối internet
- **Giải pháp:** Đảm bảo đã set `HF_HUB_OFFLINE=true` và `TRANSFORMERS_OFFLINE=true`

### Model cache ở đâu?
- **Windows:** `C:\Users\<username>\.cache\huggingface\hub\`
- **Linux/Mac:** `~/.cache/huggingface/hub/`

Bạn có thể copy thư mục cache này sang máy khác để không cần tải lại.

## Copy model sang máy khác

1. Copy thư mục cache từ máy có internet:
   ```
   ~/.cache/huggingface/hub/models--google--translategemma-4b-it/
   ```

2. Paste vào máy không có internet tại cùng đường dẫn:
   ```
   ~/.cache/huggingface/hub/models--google--translategemma-4b-it/
   ```

3. Chạy server với offline mode như hướng dẫn trên
