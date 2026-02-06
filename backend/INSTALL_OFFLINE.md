# Hướng dẫn cài đặt OFFLINE (Không có internet)

## Bước 1: Giải nén package

1. Giải nén file `.rar` hoặc `.zip` vào thư mục bất kỳ
2. Mở thư mục đã giải nén

## Bước 2: Chạy setup

### Windows:
```bash
cd backend
setup_offline.bat
```

Script này sẽ:
- Tạo virtual environment
- Cài đặt tất cả Python packages từ cache local
- Copy model cache vào đúng vị trí
- Tạo file `.env` cho offline mode

## Bước 3: Kiểm tra cài đặt

```bash
cd backend
.my-env\Scripts\activate
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## Bước 4: Chạy server

```bash
cd backend
.my-env\Scripts\activate
python main.py
```

Hoặc dùng script:
```bash
cd backend
run_offline.bat
```

## Frontend (đã build sẵn)

Trong package offline sẽ có thư mục `frontend_dist/`.

- **Cách nhanh nhất**: mở file `frontend_dist/index.html` bằng trình duyệt.
- Nếu backend không chạy trên `localhost:8000`, bạn cần build lại frontend với biến môi trường `VITE_API_URL`
  rồi đóng gói lại (vì Vite env là **build-time**).

Ví dụ:

```bash
cd frontend
set VITE_API_URL=http://<IP_BACKEND>:8000/api
npm run build
```

## Troubleshooting

### Lỗi: "Python not found"
- Cài Python 3.11+ từ python.org
- Đảm bảo Python đã được thêm vào PATH

### Lỗi: "Packages not found"
- Kiểm tra thư mục `packages/` có tồn tại không
- Nếu thiếu, bạn cần internet để cài: `pip install -r requirements.txt`

### Lỗi: "Model not found"
- Kiểm tra thư mục `model_cache/` có tồn tại không
- Nếu thiếu, bạn cần internet để tải: `python download_model.py`
- Hoặc copy model cache từ máy khác vào: `%USERPROFILE%\.cache\huggingface\hub\`

### Lỗi: "CUDA not available"
- Nếu máy có GPU: Cài CUDA toolkit và driver NVIDIA
- Nếu không có GPU: Server sẽ tự động dùng CPU

### Kiểm tra model cache location
```bash
# Windows
dir %USERPROFILE%\.cache\huggingface\hub\models--google--translategemma-4b-it

# Nếu không có, copy từ model_cache trong package
xcopy /E /I /Y model_cache\models--google--translategemma-4b-it %USERPROFILE%\.cache\huggingface\hub\models--google--translategemma-4b-it
```

## Cấu trúc package

```
csv-translator-offline/
├── backend/              # Source code
├── frontend/             # Frontend code
├── packages/             # Python packages (wheels)
├── model_cache/          # Hugging Face model cache
├── requirements-full.txt # Full package list
├── setup.bat            # Setup script
└── INSTALL.md           # This file
```

## Yêu cầu hệ thống

- **Python 3.11+** (chưa có trong package, cần cài riêng)
- **Windows 10/11** hoặc Linux
- **RAM:** Tối thiểu 8GB, khuyến nghị 16GB
- **Disk:** ~15GB dung lượng trống
- **GPU (tùy chọn):** NVIDIA GPU với CUDA support

## Lưu ý quan trọng

1. **Python không có trong package** - Cần cài Python 3.11+ trước
2. **CUDA toolkit** - Nếu dùng GPU, cần cài CUDA toolkit riêng
3. **Model cache** - Nếu thiếu, cần ~10GB để tải model
4. **Offline mode** - File `.env` đã được tạo với `HF_HUB_OFFLINE=true`

## Nếu thiếu gì đó

Nếu package thiếu một số files, bạn vẫn có thể:

1. **Cài packages từ internet:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Tải model từ internet:**
   ```bash
   python download_model.py
   ```

3. **Copy từ máy khác:**
   - Copy virtual environment: `.my-env/`
   - Copy model cache: `%USERPROFILE%\.cache\huggingface\hub/`
