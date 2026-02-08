# 🐳 TranslateGemma Docker Quick Start

## ⚡ Khởi động nhanh (30 giây)

```bash
# 1. Một lệnh để khởi động tất cả (GPU 2)
bash docker-start.sh 2

# 2. Truy cập:
# - Frontend: http://localhost
# - API: http://localhost:8000/docs
```

## 📋 Yêu cầu

Đã cài đặt:
- ✅ Docker & Docker Compose
- ✅ NVIDIA Docker Runtime
- ✅ GPU NVIDIA (A100)
- ✅ CUDA 13.1+

## 🚀 3 Cách chạy

### Cách 1: Script (Khuyến nghị)
```bash
# GPU 2 (mặc định, ~79GB free)
bash docker-start.sh 2

# Hoặc GPU khác
bash docker-start.sh 0
bash docker-start.sh 1
bash docker-start.sh 3
```

### Cách 2: Docker Compose trực tiếp
```bash
# Chạy với GPU 2
GPU_DEVICE_ID=2 CUDA_VISIBLE_DEVICES=2 docker compose up -d

# Chạy với GPU 0
GPU_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 docker compose up -d
```

### Cách 3: Sửa .env.docker
```bash
# 1. Sửa .env.docker
vim .env.docker
# GPU_DEVICE_ID=2  <- thay đổi số này

# 2. Khởi động
docker compose up -d
```

## 📍 Địa chỉ truy cập

| Dịch vụ | Địa chỉ | Mô tả |
|---------|---------|-------|
| **Frontend** | http://localhost | Giao diện web React |
| **Backend API** | http://localhost:8000 | REST API |
| **API Documentation** | http://localhost:8000/docs | Swagger UI |
| **GPU Info** | `docker exec translate-gemma-backend nvidia-smi` | Kiểm tra GPU |

## 🎮 Kinh nghiêm quý báu về GPU 2

Bạn có **4 GPUs A100 80GB**:

```
GPU 0: 75,499 MiB used / 81,920 MiB total  ❌ (93% sử dụng)
GPU 1: 20,433 MiB used / 81,920 MiB total  ⚠️  (25% sử dụng)
GPU 2:  2,335 MiB used / 81,920 MiB total  ✅ (3% sử dụng)  ← BEST
GPU 3: 21,921 MiB used / 81,920 MiB total  ⚠️  (27% sử dụng)
```

**GPU 2 có ~79GB available - đây là lựa chọn tối ưu!**

## 📊 Kiểm tra trạng thái

```bash
# Xem status các container
docker compose ps

# Xem logs backend (GPU loading, model...)
docker compose logs -f backend

# Xem logs frontend
docker compose logs -f frontend

# Kiểm tra GPU đang sử dụng
docker exec translate-gemma-backend nvidia-smi

# Theo dõi real-time
watch -n 1 'docker exec translate-gemma-backend nvidia-smi'
```

## 🛑 Dừng dịch vụ

```bash
# Cách nhanh nhất
bash docker-stop.sh

# Hoặc dùng docker compose
docker compose down

# Nếu muốn xóa tất cả dữ liệu (cảnh báo!)
docker compose down -v
```

## 🔧 Troubleshooting

### ❌ GPU không được nhận diện

```bash
# Kiểm tra NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:12.4.1-base nvidia-smi

# Nếu lỗi, cài đặt NVIDIA Container Toolkit
curl https://get.docker.com | bash

sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### ❌ Out of Memory

```bash
# Nếu GPU 2 full, thử GPU 1 (25% sử dụng)
GPU_DEVICE_ID=1 CUDA_VISIBLE_DEVICES=1 docker compose up -d

# Hoặc tăng memory limit
docker compose down
# Sửa docker-compose.yml: mem_limit: 256g
docker compose up -d
```

### ❌ Model không download được

```bash
# Cách 1: Download trước trên host
cd backend
python3 download_model.py

# Cách 2: Dùng cached model từ host
# Sửa docker-compose.yml:
# volumes:
#   - ~/.cache/huggingface:/app/model_cache
docker compose up -d
```

### ❌ Port đã bị sử dụng

```bash
# Tìm process chiếm port 80 hoặc 8000
sudo lsof -i :80
sudo lsof -i :8000

# Kill process hoặc thay port trong docker-compose.yml
```

## 📈 Monitor GPU và Memory

```bash
# Xem memory GPU khả dụng (trực tiếp)
docker exec translate-gemma-backend nvidia-smi --query-gpu=memory.free --format=csv

# Xem CPU/Memory của container
docker stats translate-gemma-backend

# Follow logs với timestamp
docker compose logs --timestamps -f backend
```

## 🔐 Cấu hình Hugging Face Token (tùy chọn)

```bash
# 1. Tạo .env file
cat > .env << EOF
GPU_DEVICE_ID=2
CUDA_VISIBLE_DEVICES=2
HF_TOKEN=hf_xxxxxxxxxxxxx
EOF

# 2. Khởi động
docker compose up -d
```

## 📦 Build lại images

Nếu bạn sửa code:

```bash
# Build lại
docker compose build --no-cache

# Hoặc xóa images và build lại
docker rmi translate-gemma:latest-backend translate-gemma:latest-frontend
docker compose build
```

## 🌐 Chạy trên Host khác (Production)

1. **Copy code**
```bash
scp -r translate-gemma user@remote-host:/opt/
```

2. **SSH vào host**
```bash
ssh user@remote-host
cd /opt/translate-gemma
```

3. **Khởi động với GPU khác (nếu cần)**
```bash
bash docker-start.sh 3  # Hoặc số GPU khác
```

## 💡 Tips & Tricks

```bash
# Khởi động lại container
docker compose restart backend

# Chạy lệnh trong container
docker compose exec backend bash

# Xem tất cả volumes
docker volume ls | grep translate

# Xem kích thước volumes
docker volume inspect translate-gemma_model_cache
du -sh /var/lib/docker/volumes/translate-gemma_model_cache/_data

# Clean up unused images/volumes
docker system prune
docker volume prune
```

## 📚 Chi tiết hơn

Xem [DOCKER_DEPLOYMENT.md](./DOCKER_DEPLOYMENT.md) để có:
- Cài đặt chi tiết NVIDIA Docker
- Cấu hình HTTPS/Nginx
- Performance tuning
- Build production images
- Và nhiều hơn nữa...

## 🆘 Cần giúp thêm?

```bash
# Xem full logs
docker compose logs backend | tail -200

# Xem error từ model loading
docker compose logs backend | grep -i "error\|failed\|exception"

# Kiểm tra kết nối API
curl http://localhost:8000/api/health
```

---

**Mẹo:** Nếu muốn dùng GPU khác, thay `2` bằng số GPU cần dùng (0, 1, 3, ...) ở mọi chỗ.
