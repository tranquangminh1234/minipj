import json
import matplotlib.pyplot as plt
import os
import pandas as pd

# Hàm đọc độ chính xác từ file JSON
def get_acc(path):
    if not os.path.exists(path):
        return 0
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            # Xử lý các định dạng khác nhau của file json
            if 'accuracy' in data: return data['accuracy']
            if 'self_training_accuracy' in data: return data['self_training_accuracy']
            if 'co_training_accuracy' in data: return data['co_training_accuracy']
            return 0
    except:
        return 0

# 1. Đọc dữ liệu từ các file kết quả đã chạy
acc_base = get_acc('data/processed/metrics.json')
acc_self = get_acc('data/processed/metrics_self_training.json')
acc_co   = get_acc('data/processed/metrics_co_training.json')

print("\n=== KẾT QUẢ ĐỘ CHÍNH XÁC (ACCURACY) ===")
print(f"1. Supervised (Baseline): {acc_base:.4f}")
print(f"2. Self-Training:         {acc_self:.4f}")
print(f"3. Co-Training:           {acc_co:.4f}")

# 2. Vẽ biểu đồ so sánh
methods = ['Supervised\n(Baseline)', 'Self-Training', 'Co-Training']
scores = [acc_base, acc_self, acc_co]
colors = ['gray', '#1f77b4', '#2ca02c'] # Xám, Xanh dương, Xanh lá

plt.figure(figsize=(8, 5))
bars = plt.bar(methods, scores, color=colors, width=0.6)

# Thêm số lên đầu cột
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2%}", ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title('So sánh hiệu quả các phương pháp (Semi-supervised)', fontsize=14)
plt.ylabel('Độ chính xác (Accuracy)')
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# 3. Lưu ảnh
out_path = 'data/processed/FINAL_REPORT_CHART.png'
plt.savefig(out_path)
print(f"\n✅ Đã lưu biểu đồ báo cáo tại: {out_path}")
print("Đang mở ảnh lên cho bạn xem...")
plt.show()