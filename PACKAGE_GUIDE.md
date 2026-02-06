# Hướng dẫn đóng gói và cài đặt OFFLINE

## 📦 Bước 1: Đóng gói trên máy CÓ INTERNET

### Windows:
```bash
package_for_offline.bat
```

### Linux/Mac:
```bash
chmod +x package_for_offline.sh
./package_for_offline.sh
```

Script sẽ tạo thư mục `csv-translator-offline-YYYYMMDD` chứa:
- ✅ Toàn bộ source code (backend + frontend)
- ✅ Python packages đã download (wheels)
- ✅ Model cache (~10GB)
- ✅ Scripts setup tự động
- ✅ Hướng dẫn cài đặt

### Sau khi chạy script:

1. **Kiểm tra package:**
   ```
   csv-translator-offline-YYYYMMDD/
   ├── backend/
   ├── frontend/
   ├── packages/          (Python wheels)
   ├── model_cache/       (Model files ~10GB)
   ├── requirements-full.txt
   ├── setup.bat (hoặc setup.sh)
   └── INSTALL.md
   ```

2. **Nén thành file .rar hoặc .zip:**
   - Chuột phải vào thư mục → Send to → Compressed folder
   - Hoặc dùng WinRAR/7-Zip

3. **Copy file .rar sang máy không có internet**

## 🚀 Bước 2: Cài đặt trên máy KHÔNG CÓ INTERNET

### Yêu cầu trước:
- ✅ Python 3.11+ đã cài (download từ python.org)
- ✅ Đủ dung lượng ổ đĩa (~15GB)

### Các bước:

1. **Giải nén file .rar:**
   - Giải nén vào thư mục bất kỳ (ví dụ: `C:\csv-translator\`)

2. **Chạy setup:**
   ```bash
   cd csv-translator-offline-YYYYMMDD\backend
   setup.bat
   ```

3. **Chạy server:**
   ```bash
   cd backend
   .my-env\Scripts\activate
   python main.py
   ```

   Hoặc dùng script:
   ```bash
   run_offline.bat
   ```

## ⚠️ Lưu ý quan trọng

### 1. Python không có trong package
- **Phải cài Python 3.11+ trước** trên máy đích
- Download từ: https://www.python.org/downloads/
- Đảm bảo check "Add Python to PATH" khi cài

### 2. CUDA (nếu dùng GPU)
- Nếu máy đích có GPU NVIDIA:
  - Cài CUDA toolkit (version 12.4+)
  - Cài NVIDIA driver mới nhất
- Nếu không có GPU: Server tự động dùng CPU

### 3. Model cache
- Nếu package có `model_cache/`: ✅ Đã sẵn sàng
- Nếu thiếu: Cần internet để tải (~10GB)

### 4. Packages
- Nếu package có `packages/`: ✅ Cài từ cache
- Nếu thiếu một số: Cần internet để cài thêm

## 🔧 Troubleshooting

### Lỗi: "Python not found"
```bash
# Kiểm tra Python đã cài chưa
python --version

# Nếu không có, cài Python 3.11+ từ python.org
```

### Lỗi: "Packages installation failed"
```bash
# Thử cài từ requirements-full.txt
pip install --no-index --find-links=packages -r requirements-full.txt

# Hoặc cài từng package thủ công
pip install --no-index --find-links=packages package_name
```

### Lỗi: "Model not found"
```bash
# Kiểm tra model cache
dir %USERPROFILE%\.cache\huggingface\hub\models--google--translategemma-4b-it

# Nếu thiếu, copy từ package
xcopy /E /I /Y model_cache\models--google--translategemma-4b-it %USERPROFILE%\.cache\huggingface\hub\models--google--translategemma-4b-it
```

### Lỗi: "CUDA not available"
- Không có GPU: Bình thường, server sẽ dùng CPU
- Có GPU nhưng lỗi: Cài CUDA toolkit và driver NVIDIA

## 📋 Checklist trước khi đóng gói

- [ ] Đã chạy `python download_model.py` để tải model
- [ ] Virtual environment đã có và hoạt động
- [ ] Đã test server chạy được
- [ ] Đã có đủ dung lượng (~15GB)

## 📋 Checklist trước khi cài đặt

- [ ] Python 3.11+ đã cài
- [ ] Đủ dung lượng ổ đĩa (~15GB)
- [ ] Đã giải nén package
- [ ] Đã đọc file INSTALL.md

## 💡 Tips

1. **Test package trước khi copy:**
   - Giải nén trên máy nguồn
   - Chạy `setup.bat` để test
   - Nếu OK thì mới copy sang máy đích

2. **Nếu package quá lớn:**
   - Có thể tách riêng `model_cache/` (copy sau)
   - Hoặc dùng external drive

3. **Nếu thiếu gì:**
   - Copy thư mục `.my-env` từ máy nguồn
   - Copy model cache từ máy nguồn
   - Hoặc cài thủ công khi có internet
