# CSV Translator Pro

**Phần mềm dịch thuật AI offline sử dụng mô hình TranslateGemma của Google**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Tổng quan

CSV Translator Pro là phần mềm dịch thuật AI **hoàn toàn offline**, được thiết kế cho môi trường yêu cầu bảo mật cao. Phần mềm sử dụng mô hình **TranslateGemma** của Google để dịch thuật chất lượng cao giữa 55 ngôn ngữ.

### ✨ Tính năng chính

| Tính năng | Mô tả |
|-----------|-------|
| 📄 **Dịch file CSV hàng loạt** | Upload CSV → Chọn cột → Dịch tự động → Tải kết quả |
| 💬 **Dịch văn bản trực tiếp** | Nhập text → Dịch ngay lập tức |
| 🖼️ **Dịch từ ảnh (OCR)** | Upload ảnh → Nhận dạng chữ → Dịch tự động |
| 🔒 **100% Offline** | Dữ liệu không rời khỏi mạng nội bộ |
| 🌍 **55 ngôn ngữ** | Hỗ trợ đặc biệt tốt với tiếng Ả Rập |

## 🏗️ Kiến trúc hệ thống

```
┌──────────────┐     HTTP/REST      ┌──────────────┐     Inference     ┌──────────────┐
│   CLIENTS    │ ◄──────────────► │    SERVER    │ ◄──────────────► │   AI MODEL   │
│  (Browser)   │      API          │   FastAPI    │                   │ Gemma-27B    │
│  1000 users  │                   │  Python 3.12 │                   │ PyTorch+CUDA │
└──────────────┘                   └──────────────┘                   └──────┬───────┘
                                                                              │
                                                                              ▼
                                                                      ┌──────────────┐
                                                                      │  GPU Server  │
                                                                      │ A100 80GB    │
                                                                      └──────────────┘
```

## 🚀 Cài đặt

### Yêu cầu hệ thống

**Server:**
- GPU: NVIDIA A100 80GB (khuyến nghị) hoặc RTX 4090 24GB (tối thiểu)
- CPU: AMD EPYC / Intel Xeon (16+ cores)
- RAM: 128 GB DDR4 ECC
- Storage: NVMe SSD 2TB
- OS: Ubuntu 22.04 LTS / Windows Server 2022

**Client:**
- Trình duyệt: Chrome / Edge / Firefox (phiên bản mới nhất)
- Kết nối: LAN/WiFi đến Server

### Cài đặt Backend

```bash
cd backend

# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy server
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Cài đặt Frontend

```bash
cd frontend

# Cài đặt dependencies
npm install

# Build production
npm run build

# Hoặc chạy development
npm run dev
```

## 📡 API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/api/languages` | Danh sách 55 ngôn ngữ |
| POST | `/api/translate-text` | Dịch văn bản |
| POST | `/api/translate-image` | OCR + Dịch từ ảnh |
| POST | `/api/upload` | Upload CSV để dịch |
| GET | `/api/status/{job_id}` | Kiểm tra tiến trình |
| GET | `/api/download/{job_id}` | Tải file đã dịch |
| GET | `/api/health` | Kiểm tra server |

## 🤖 Mô hình AI

### TranslateGemma-27B-IT

| Thông số | Giá trị |
|----------|---------|
| Tham số | 27 tỷ (27B) |
| Kiến trúc | Gemma 3 Decoder-only |
| Ngôn ngữ | 55 |
| Context | 2,048 tokens |
| VRAM | 58-62 GB (BF16) |

### Quantization Options

| Phương pháp | VRAM | Chất lượng | GPU tối thiểu |
|-------------|------|------------|---------------|
| BF16 (Full) | ~60 GB | ⭐⭐⭐⭐⭐ | A100 80GB |
| INT8 | ~32 GB | ⭐⭐⭐⭐ | A6000 48GB |
| NF4 (4-bit) | ~20 GB | ⭐⭐⭐ | RTX 4090 24GB |

## 📁 Cấu trúc dự án

```
csv-translator-pro/
├── backend/                 # FastAPI server
│   ├── main.py             # Entry point
│   ├── gemma_translator.py # TranslateGemma wrapper
│   ├── requirements.txt    # Python dependencies
│   └── uploads/            # Uploaded files
├── frontend/               # React application
│   ├── src/
│   │   ├── App.jsx        # Main component
│   │   └── components/    # UI components
│   └── package.json
├── docs/                   # Tài liệu kỹ thuật
│   ├── bao_cao_phan_cung_phan_mem.pdf
│   ├── bao_cao_tinh_nang_ky_thuat.pdf
│   └── ke_hoach_trien_khai.pdf
└── README.md
```

## 📊 Hiệu năng

| Số dòng CSV | Thời gian (A100 80GB) | Tokens/giây |
|-------------|----------------------|-------------|
| 100 | 2-3 phút | 40-70 |
| 500 | 8-12 phút | 40-70 |
| 1,000 | 15-22 phút | 40-70 |
| 5,000 | 1-2 giờ | 40-70 |

## 🔐 Bảo mật

- ✅ **Hoạt động 100% offline** sau khi cài đặt
- ✅ **Dữ liệu nội bộ** - không gửi ra Internet
- ✅ **Không lưu log nội dung** - chỉ log kỹ thuật
- ✅ **Xóa tự động** - file tạm xóa sau 7 ngày

## 📚 Tài liệu

- [Báo cáo phần cứng & phần mềm](docs/bao_cao_phan_cung_phan_mem.pdf)
- [Báo cáo tính năng kỹ thuật](docs/bao_cao_tinh_nang_ky_thuat.pdf)
- [Kế hoạch triển khai](docs/ke_hoach_trien_khai.pdf)
- [Hướng dẫn đóng gói offline](PACKAGE_GUIDE.md)

## 🛠️ Phát triển

```bash
# Clone repository
git clone https://github.com/Phambanam99/translate-gemma.git
cd translate-gemma

# Chạy backend (development)
cd backend
python -m uvicorn main:app --reload --port 8000

# Chạy frontend (development)
cd frontend
npm run dev
```

## 📄 License

MIT License - Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

## 👥 Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng tạo Issue hoặc Pull Request.

---

**CSV Translator Pro** - Dịch thuật AI offline, bảo mật tuyệt đối 🔒
