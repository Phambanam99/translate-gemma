# Hướng dẫn cấu hình Biến Môi Trường và CUDA

## 1. Biến Môi Trường

### Tạo file `.env`

Copy file `.env.example` thành `.env`:

```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

Sau đó chỉnh sửa file `.env` theo nhu cầu.

### Các biến môi trường quan trọng

#### Offline Mode
```env
HF_HUB_OFFLINE=true          # Bật chế độ offline (không cần internet)
TRANSFORMERS_OFFLINE=true    # Tương tự cho transformers
```

#### CUDA Settings
```env
CUDA_VISIBLE_DEVICES=0       # Chỉ dùng GPU 0 (hoặc 1, 2, ...)
CUDA_LAUNCH_BLOCKING=1       # Debug CUDA errors (chậm hơn)
```

#### Cache Directory
```env
HF_HOME=C:\Users\YourName\.cache\huggingface
TRANSFORMERS_CACHE=C:\Users\YourName\.cache\huggingface
```

## 2. Cài đặt CUDA

### Kiểm tra CUDA đã cài chưa

```bash
# Kiểm tra CUDA version
nvcc --version

# Kiểm tra GPU
nvidia-smi
```

### Cài đặt PyTorch với CUDA

**Windows (CUDA 12.4):**
```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

**Linux (CUDA 12.4):**
```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

**CPU only (không có GPU):**
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
```

### Kiểm tra PyTorch có CUDA không

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

## 3. Scripts chạy với cấu hình khác nhau

### Chạy với .env file

Server sẽ tự động load `.env` khi khởi động:

```bash
python main.py
```

### Chạy offline (Windows)

```bash
run_offline.bat
```

Hoặc thủ công:
```powershell
$env:HF_HUB_OFFLINE="true"
$env:TRANSFORMERS_OFFLINE="true"
python main.py
```

### Chạy với GPU cụ thể

```powershell
# Chỉ dùng GPU 0
$env:CUDA_VISIBLE_DEVICES="0"
python main.py

# Dùng GPU 1
$env:CUDA_VISIBLE_DEVICES="1"
python main.py
```

### Chạy trên CPU (không dùng GPU)

```powershell
$env:CUDA_VISIBLE_DEVICES=""
python main.py
```

## 4. Troubleshooting

### Lỗi: "CUDA out of memory"

**Giải pháp:**
1. Model tự động dùng 4-bit quantization trên GPU < 12GB
2. Hoặc set trong `.env`:
   ```env
   MODEL_QUANTIZATION=4bit
   ```

### Lỗi: "CUDA not available"

**Kiểm tra:**
1. Đã cài CUDA toolkit chưa?
2. PyTorch có build với CUDA không?
3. GPU driver đã cài đúng chưa?

**Test:**
```python
import torch
print(torch.cuda.is_available())  # Phải là True
```

### Lỗi: "Model not found" khi offline

**Giải pháp:**
1. Chạy `python download_model.py` khi có internet
2. Hoặc copy model cache từ máy khác

### Lỗi: "bitsandbytes not found"

**Giải pháp:**
```bash
pip install bitsandbytes
```

**Lưu ý:** bitsandbytes chỉ hoạt động trên Linux. Trên Windows, model sẽ tự động fallback về CPU hoặc float16.

## 5. Requirements.txt và CUDA

File `requirements.txt` hiện tại:
- `torch==2.6.0` - Sẽ cài CPU version nếu không chỉ định
- Để cài CUDA version, dùng lệnh riêng (xem phần 2)

### Tạo requirements-cuda.txt (tùy chọn)

```txt
# requirements-cuda.txt
# Chạy: pip install -r requirements-cuda.txt

# Core dependencies
fastapi
uvicorn[standard]
transformers
sentencepiece
pandas
python-multipart
aiofiles
Pillow
accelerate
requests
sacremoses
packaging
huggingface_hub

# PyTorch với CUDA (cài riêng)
# pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# bitsandbytes (chỉ Linux)
# pip install bitsandbytes
```

## 6. Checklist trước khi chạy

- [ ] Đã cài CUDA toolkit (nếu dùng GPU)
- [ ] Đã cài PyTorch với CUDA support
- [ ] Đã tải model về (nếu chạy offline)
- [ ] Đã tạo file `.env` (nếu cần)
- [ ] Đã cài tất cả packages: `pip install -r requirements.txt`
- [ ] Đã test: `python -c "import torch; print(torch.cuda.is_available())"`

## 7. Ví dụ cấu hình đầy đủ

### Máy có GPU, có internet:
```env
# .env
HF_HUB_OFFLINE=false
TRANSFORMERS_OFFLINE=false
CUDA_VISIBLE_DEVICES=0
```

### Máy có GPU, không có internet:
```env
# .env
HF_HUB_OFFLINE=true
TRANSFORMERS_OFFLINE=true
CUDA_VISIBLE_DEVICES=0
```

### Máy không có GPU:
```env
# .env
HF_HUB_OFFLINE=false
TRANSFORMERS_OFFLINE=false
CUDA_VISIBLE_DEVICES=
MODEL_DEVICE=cpu
```
