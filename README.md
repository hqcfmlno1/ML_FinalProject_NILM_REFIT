# NILM REFIT - Non-Intrusive Load Monitoring Using Machine Learning

## Giới thiệu (Introduction)

Dự án này tập trung vào **Non-Intrusive Load Monitoring (NILM)** - công nghệ phân tích tiêu thụ năng lượng của các thiết bị điện trong nhà từ dữ liệu tổng công suất mains mà không cần cài đặt cảm biến riêng cho từng thiết bị.

**What is NILM?**
- NILM là kỹ thuật "nhìn" vào tín hiệu điện tổng và xác định xem các thiết bị nào đang chạy và đang tiêu thụ bao nhiêu điện
- Ứng dụng: tiết kiệm năng lượng, lập hóa đơn điện chính xác, quản lý lưới điện thông minh

## Tính năng chính (Key Features)

- Xử lý dữ liệu từ dataset REFIT  
- Các mô hình từ cơ bản đến nâng cao (Basic & Advanced Models)  
- Window shifting technique cho xử lý dữ liệu chuỗi thời gian  
- Checkpoints lưu trữ các mô hình đã huấn luyện  

## Cấu trúc dự án (Project Structure)

```
ML_FinalProject_NILM_REFIT/
├── README.md                    
├── requirements.txt             # Các thư viện cần thiết
│
├── data/                        # Dữ liệu
│   ├── processed_data/          # Dữ liệu đã xử lý
│   │   ├── House2_full.csv      # Dữ liệu đầy đủ nhà 2
│   │   ├── House2_part1-5.csv   # Dữ liệu chia thành 5 phần
│   ├── train/                   # Dữ liệu huấn luyện
│   └── test/                    # Dữ liệu kiểm tra
│
├── notebooks/                   # Jupyter Notebooks
│   ├── data_merge.ipynb         # Gộp dữ liệu từ các phần
│   └── window_shilfter_test.ipynb  # Kiểm tra kỹ thuật window shifting
│
├── src/                         # Mã nguồn
│   ├── models/                  # Định nghĩa các mô hình
│   └── tools/                   # Các công cụ tiện ích
│       └── window_shilfter.py   # Kiến trúc sliding window
│
└── checkpoints/                 # Lưu trữ mô hình
    ├── basic models/            # Các mô hình cơ bản
    └── advanced models/         # Các mô hình nâng cao
```

## Cài đặt và Thiết lập (Installation)

### Yêu cầu (Requirements)
- Python 3.8 trở lên
- Các thư viện: pandas, numpy, scikit-learn, tensorflow/pytorch, matplotlib, etc.

### Bước cài đặt

1. Clone hoặc tải xuống dự án:
```bash
cd ML_FinalProject_NILM_REFIT
```

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

3. Chuẩn bị dữ liệu:
```bash
# Chạy notebook để gộp dữ liệu
jupyter notebook notebooks/data_merge.ipynb
```

## Dữ liệu (Dataset)

**REFIT Dataset:**
- Chứa dữ liệu điện hàng giờ từ 20 gia đình tại Anh
- Bao gồm dữ liệu tổng công suất mains (aggregate) và công suất từng thiết bị (sub-metering)
- Dự án này tập trung vào **House 2** (Nhà số 2)

**Dữ liệu House 2:**
- File đầy đủ: `House2_full.csv`
- Chia thành 5 phần: `House2_part1.csv` - `House2_part5.csv`

### Cấu trúc dữ liệu
- Time: Thời gian đo (yyyy:mm:dd)
- Unix
- Aggregate: Tổng công suất tiêu thụ (bao gồm cả công suất của các thiết bị không được theo dõi)
- Appliance{i}: Công suất tiêu thụ của thiết bị i

