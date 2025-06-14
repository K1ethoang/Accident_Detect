import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from matplotlib.colors import Normalize
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

# Hàm tạo gradient dựa trên palette mặc định
def get_gradient_palette(values, cmap_name, reverse=False, clip=0.5):
    values = list(values)
    min_val, max_val = min(values), max(values)
    
    # Co vùng chuẩn hóa lại để tránh màu nhạt nhất
    vmin = min_val - (max_val - min_val) * clip
    vmax = max_val

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    colors = [cmap(norm(val)) for val in values]
    return colors[::-1] if reverse else colors

# Setup style
sns.set_theme(style='white', context='paper')
plt.figure(figsize=(16, 10))

# Biểu đồ mAP50
plt.subplot(2, 2, 1)
colors_map = get_gradient_palette(df_merged["mAP50"], "Blues")
ax1 = sns.barplot(x='mAP50', y='Version', data=df_merged, palette=colors_map)
plt.title('mAP@50 Comparison')
for container in ax1.containers:
    ax1.bar_label(container, fmt='%.4f')

# Biểu đồ IoU
plt.subplot(2, 2, 2)
colors_iou = get_gradient_palette(df_merged["Average_IoU"], "Greens")
ax2 = sns.barplot(x='Average_IoU', y='Version', data=df_merged, palette=colors_iou)
plt.title('Average IoU Comparison')
for container in ax2.containers:
    ax2.bar_label(container, fmt='%.4f')

# Biểu đồ FPS
plt.subplot(2, 2, 3)
colors_fps = get_gradient_palette(df_merged["FPS"], "Oranges")
ax3 = sns.barplot(x='FPS', y='Version', data=df_merged, palette=colors_fps)
plt.title('FPS (1000 frames) Comparison')
for container in ax3.containers:
    ax3.bar_label(container, fmt='%.2f')

# Biểu đồ Latency
plt.subplot(2, 2, 4)
colors_latency = get_gradient_palette(df_merged["Latency (s)"], "Reds")
ax4 = sns.barplot(x='Latency (s)', y='Version', data=df_merged, palette=colors_latency)
plt.title('Latency (s) Comparison')
for container in ax4.containers:
    ax4.bar_label(container, fmt='%.4f')

plt.suptitle("YOLOv11 Performance Comparison time 3", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(output_image, dpi=300)
plt.show()
