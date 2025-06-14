import os
import pandas as pd
import re
import matplotlib.pyplot as plt

# Root folder chứa các thư mục time_1, time_2, ...
ROOT_DIR = "./evaluate"
LATENCY_FILE = "latency_1k_frame.csv"
MAP_IOU_FILE = "mAP50_IoU.csv"

# Tự động tìm thư mục có tên dạng time_<n>
run_folders = sorted([
    f for f in os.listdir(ROOT_DIR)
    if os.path.isdir(os.path.join(ROOT_DIR, f)) and re.match(r"time_\d+", f)
])

# Đọc dữ liệu
data = []
for run in run_folders:
    map_path = os.path.join(ROOT_DIR, run, MAP_IOU_FILE)
    latency_path = os.path.join(ROOT_DIR, run, LATENCY_FILE)

    if not os.path.exists(map_path) or not os.path.exists(latency_path):
        continue

    df_map = pd.read_csv(map_path)
    df_latency = pd.read_csv(latency_path)

    for version in df_map["Version"]:
        data.append({"Metric": "mAP50", "Version": version, run: df_map[df_map["Version"] == version]["mAP50"].values[0]})
        data.append({"Metric": "Average_IoU", "Version": version, run: df_map[df_map["Version"] == version]["Average_IoU"].values[0]})
        data.append({"Metric": "FPS", "Version": version, run: df_latency[df_latency["Version"] == version]["FPS"].values[0]})
        data.append({"Metric": "Latency (s)", "Version": version, run: df_latency[df_latency["Version"] == version]["Latency (s)"].values[0]})

# Tạo dataframe và pivot
df = pd.DataFrame(data)
pivot_df = df.pivot_table(index=["Metric", "Version"], aggfunc="first")

# Sắp xếp version theo thứ tự custom trong từng group Metric
custom_version_order = ["v11-n", "v11-s", "v11-m", "v11-l", "v1-x"]
version_rank = {v: i for i, v in enumerate(custom_version_order)}

# Sắp xếp trong từng nhóm Metric
pivot_df = pivot_df.reset_index()
pivot_df = pivot_df.groupby("Metric", group_keys=False).apply(
    lambda group: group.sort_values(by="Version", key=lambda col: col.map(version_rank))
)
pivot_df = pivot_df.set_index(["Metric", "Version"])

# Xuất Excel
output_excel = os.path.join(ROOT_DIR, "summary_new.xlsx")
pivot_df.to_excel(output_excel)
print(f"[+] Saved to: {output_excel}")

# Vẽ biểu đồ line cho từng metric
output_chart_dir = os.path.join(ROOT_DIR, "charts")
os.makedirs(output_chart_dir, exist_ok=True)

for metric in pivot_df.index.get_level_values("Metric").unique():
    metric_df = pivot_df.loc[metric]
    metric_df = metric_df.loc[metric_df.index.intersection(custom_version_order)]  # giữ thứ tự đúng

    metric_df = metric_df.reindex(custom_version_order).dropna(how='all')

    plt.figure(figsize=(10, 6))
    for col in metric_df.columns:
        plt.plot(metric_df.index, metric_df[col], marker='o', label=col)

    plt.title(f"{metric} over runs")
    plt.xlabel("Version")
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend(title="Run")
    plt.tight_layout()
    chart_path = os.path.join(output_chart_dir, f"{metric.replace(' ', '_')}.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"[+] Saved chart: {chart_path}")
