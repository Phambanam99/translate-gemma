# Docker Deployment Guide for TranslateGemma with GPU 2

## Yêu cầu

1. **Docker & Docker Compose**
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install -y docker.io docker-compose-plugin
   sudo usermod -aG docker $USER
   
   # Restart Docker daemon
   sudo systemctl restart docker
   ```

2. **NVIDIA Docker Runtime** (để sử dụng GPU trong container)
   ```bash
   # Cài đặt NVIDIA Docker
   curl https://get.docker.com | bash
   
   # Cài NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

3. **Kiểm tra NVIDIA Docker**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.1-base nvidia-smi
   ```

## Chạy Project với Docker

### 1. Build Images

```bash
# Chạy từ thư mục root của project
docker compose build

# Hoặc build riêng lẻ
docker build -t translate-gemma:backend ./backend
docker build -t translate-gemma:frontend ./frontend
```

### 2. Chạy các Container

```bash
# Khởi động tất cả services (backend + frontend)
docker compose up -d

# Xem logs
docker compose logs -f backend
docker compose logs -f frontend

# Dừng services
docker compose down
```

### 3. Truy cập ứng dụng

- **Frontend**: http://localhost (port 80)
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Sử dụng GPU 2

### Option 1: Sử dụng docker-compose.yml mặc định

File `docker-compose.yml` đã được cấu hình để sử dụng **GPU 2** bằng cách:

```yaml
environment:
  - GPU_DEVICE_ID=2
  - CUDA_VISIBLE_DEVICES=2
```

### Option 2: Thay đổi GPU Device

Chỉnh sửa `docker-compose.yml` hoặc chạy với override:

```bash
# Sử dụng GPU 0
GPU_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 docker compose up -d

# Sử dụng GPU 3
GPU_DEVICE_ID=3 CUDA_VISIBLE_DEVICES=3 docker compose up -d
```

### Option 3: Tạo file `.env` để cấu hình

Tạo file `.env` tại thư mục root:

```env
# GPU Configuration
GPU_DEVICE_ID=2
CUDA_VISIBLE_DEVICES=2

# Model Cache
TRANSFORMERS_CACHE=/app/model_cache
HF_HOME=/app/model_cache
```

Khởi động:

```bash
docker compose up -d
```

## Kiểm tra GPU Được Sử Dụng

```bash
# Xem GPU info của container đang chạy
docker exec translate-gemma-backend nvidia-smi

# Xem logs backend
docker exec translate-gemma-backend tail -f /tmp/backend.log
```

## Volume & Persistent Data

Docker Compose sử dụng named volumes để lưu trữ dữ liệu:

```bash
# Xem volumes
docker volume ls | grep translate

# Xem đường dẫn trên host
docker volume inspect translate-gemma_model_cache
```

**Các volumes:**
- `translate-gemma_model_cache`: Cache mô hình AI (lớn ~20GB)
- `translate-gemma_uploads`: File CSV được upload
- `translate-gemma_outputs`: File CSV đã dịch

## Troubleshooting

### 1. GPU không được nhận diện

```bash
# Kiểm tra NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:12.4.1-base nvidia-smi

# Xem logs container
docker compose logs backend | grep -i gpu

# Kiểm tra CUDA_VISIBLE_DEVICES
docker exec translate-gemma-backend bash -c 'echo $CUDA_VISIBLE_DEVICES'
```

### 2. Model không download được

Đầu tiên download model trên host:

```bash
cd backend
python3 download_model.py
```

Hoặc mount thư mục cache:

```yaml
volumes:
  - ~/.cache/huggingface:/app/model_cache
```

### 3. Out of Memory

- **Tăng memory limit** trong docker-compose.yml:
```yaml
mem_limit: 256g
shm_size: 32gb
```

- **Sử dụng GPU khác** (nếu GPU 2 có memory ít hơn)

## Build Production Image

```bash
# Backend
docker build -t myregistry/translate-gemma:latest-backend ./backend
docker push myregistry/translate-gemma:latest-backend

# Frontend  
docker build -t myregistry/translate-gemma:latest-frontend ./frontend
docker push myregistry/translate-gemma:latest-frontend
```

## Docker Compose Commands

```bash
# Khởi động services
docker compose up -d

# Dừng services
docker compose down

# Xem status
docker compose ps

# Xem logs tất cả
docker compose logs -f

# Xem logs 1 service
docker compose logs -f backend

# Chạy lệnh trong container
docker compose exec backend bash

# Rebuild và khởi động lại
docker compose up -d --build

# Xóa volumes (cảnh báo: mất dữ liệu!)
docker compose down -v
```

## Cấu hình Nginx (HTTPS)

Nếu muốn thêm HTTPS, tạo file `frontend/nginx.conf`:

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/nginx/certs/cert.pem;
    ssl_certificate_key /etc/nginx/certs/key.pem;
    
    # ... rest of config
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

Cập nhật `docker-compose.yml`:

```yaml
frontend:
  volumes:
    - ./certs:/etc/nginx/certs:ro
    - ./frontend/nginx.conf:/etc/nginx/conf.d/default.conf:ro
```

## Performance Tuning

### Tối ưu hóa GPU Memory

```yaml
environment:
  - CUDA_LAUNCH_BLOCKING=1
  - CUDA_DEVICE_ORDER=PCI_BUS_ID
```

### Tăng Performance

```bash
# Chạy container với pinned memory
docker compose up -d --memory 200g --memory-swap 200g
```

## Monitoring

```bash
# Xem resource usage
docker stats translate-gemma-backend

# Xem GPU usage
docker exec translate-gemma-backend nvidia-smi

# Real-time monitoring
watch -n 1 'docker stats translate-gemma-backend'
```

---

**Lưu ý:** Thay đổi `GPU_DEVICE_ID=2` nếu muốn sử dụng GPU khác.
