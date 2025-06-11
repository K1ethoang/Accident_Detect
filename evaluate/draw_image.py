import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# File paths
map_file = "./evaluate/mAP50_IoU.csv"
latency_file = "./evaluate/latency_1k_frame.csv"
output_image = "./evaluate/yolov11_comparison.png"

# Đọc dữ liệu
df_map = pd.read_csv(map_file)
df_latency = pd.read_csv(latency_file)

# Chỉ lấy weight là best.pt
df_map_best = df_map[df_map["Weight"] == "best.pt"].copy()

# Gộp dữ liệu theo Version
df_merged = pd.merge(df_map_best, df_latency, on="Version")

# Setup style
sns.set_theme(style='whitegrid')
plt.figure(figsize=(14, 10))

# Biểu đồ mAP50
plt.subplot(2, 2, 1)
ax1 = sns.barplot(x='mAP50', y='Version', data=df_merged, palette='Blues_d')
plt.title('mAP@50 Comparison by Version')
for container in ax1.containers:
    ax1.bar_label(container, fmt='%.4f')

# Biểu đồ Average IoU
plt.subplot(2, 2, 2)
ax2 = sns.barplot(x='Average_IoU', y='Version', data=df_merged, palette='Greens_d')
plt.title('Average IoU Comparison by Version')
for container in ax2.containers:
    ax2.bar_label(container, fmt='%.4f')

# Biểu đồ FPS
plt.subplot(2, 2, 3)
ax3 = sns.barplot(x='FPS', y='Version', data=df_merged, palette='Oranges_d')
plt.title('FPS (1000 frames) by Version')
for container in ax3.containers:
    ax3.bar_label(container, fmt='%.2f')

# Biểu đồ Latency
plt.subplot(2, 2, 4)
ax4 = sns.barplot(x='Latency (s)', y='Version', data=df_merged, palette='Reds_d')
plt.title('Latency (s) per Frame by Version')
for container in ax4.containers:
    ax4.bar_label(container, fmt='%.4f')

plt.suptitle("YOLOv11 Performance Comparison", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Lưu hình ra file
plt.savefig(output_image, dpi=300)
plt.show()
