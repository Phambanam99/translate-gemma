# Hướng dẫn setup cho NVIDIA A4000 16GB với CUDA 12.7

## Thông tin hệ thống

- **GPU:** NVIDIA A4000
- **VRAM:** 16GB
- **CUDA:** 12.7
- **Tối ưu:** Model sẽ chạy ở float16 (không cần quantization)

## Bước 1: Kiểm tra CUDA và GPU

```bash
# Kiểm tra CUDA version
nvcc --version

# Kiểm tra GPU
nvidia-smi
```

Bạn sẽ thấy:
```
NVIDIA A4000
Driver Version: 566.xx (hoặc tương tự)
CUDA Version: 12.7
```

## Bước 2: Cài đặt PyTorch với CUDA

### Nếu có internet:
```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

### Nếu không có internet (từ package):
```bash
cd backend
setup_gpu.bat
```

Hoặc thủ công:
```bash
pip install --no-index --find-links=../packages torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124
```

## Bước 3: Verify installation

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0)); print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), 'GB')"
```

Kết quả mong đợi:
```
CUDA available: True
GPU: NVIDIA A4000
VRAM: 16.0 GB
```

## Bước 4: Chạy server

```bash
cd backend
.my-env\Scripts\activate
python main.py
```

## Tối ưu cho A4000 16GB

Với 16GB VRAM, model TranslateGemma 4B sẽ:
- ✅ Chạy ở **float16** (không quantization)
- ✅ **Tốc độ nhanh nhất** (không bị giảm do quantization)
- ✅ **Chất lượng tốt nhất** (không bị mất thông tin)
- ✅ **VRAM sử dụng:** ~8GB (còn dư 8GB cho batch processing)

## Performance

- **Model size:** ~8GB VRAM (float16)
- **Inference speed:** ~10-20 tokens/second
- **Batch processing:** Có thể xử lý nhiều requests đồng thời
- **Memory headroom:** 8GB dư cho caching và batching

## Troubleshooting

### Lỗi: "CUDA version mismatch"
- CUDA 12.7 driver tương thích với PyTorch CUDA 12.4
- Không cần cài lại CUDA toolkit

### Lỗi: "Out of memory"
- Với 16GB VRAM, không nên xảy ra
- Nếu có, kiểm tra các process khác đang dùng GPU:
  ```bash
  nvidia-smi
  ```

### Lỗi: "CUDA not available"
1. Kiểm tra driver: `nvidia-smi`
2. Kiểm tra PyTorch: `python -c "import torch; print(torch.cuda.is_available())"`
3. Nếu False, cài lại PyTorch với CUDA

## So sánh với các GPU khác

| GPU | VRAM | Quantization | Performance |
|-----|------|--------------|-------------|
| RTX 3060 Ti | 8GB | 4-bit | Chậm hơn, chất lượng giảm |
| RTX 3090 | 24GB | None (float16) | Tương tự A4000 |
| **A4000** | **16GB** | **None (float16)** | **Tối ưu** |

## Tips

1. **Batch size:** Có thể tăng batch size để xử lý nhiều text cùng lúc
2. **Concurrent requests:** Server có thể xử lý nhiều requests song song
3. **Model caching:** Model sẽ cache trong VRAM, không cần load lại mỗi lần
