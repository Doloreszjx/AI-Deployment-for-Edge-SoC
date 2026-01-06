import time
import psutil
from ultralytics import YOLO
import os
from datetime import datetime

# 强制使用CPU
# 这是模型在未做 SoC 适配、未量化、未算子优化前的原始性能，用作端侧部署对比基线
baseLine_version = "yolov8n.pt"
model = YOLO(baseLine_version)

# 预热
for _ in range(5):
    model("https://ultralytics.com/images/bus.jpg", device="cpu")

# 开始测试，记录开始时间和起始已使用的物理内存（用作计算测试所占内存）
start_memory = psutil.Process().memory_info().rss / 1024 / 1024
start_time = time.time()

results = model("https://ultralytics.com/images/bus.jpg", device="cpu")

end_time = time.time()
end_memory = psutil.Process().memory_info().rss / 1024 / 1024

print(f"Interface Time: {(end_time - start_time) * 1000: .2f} ms")
print(f"Memory Usage: {end_memory - start_memory: .2f} MB")

data_cache_dir = 'data_cache'
os.makedirs(data_cache_dir, exist_ok=True)

file_name = os.path.join(data_cache_dir, 'baseline_infos.txt')
date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
with open(file_name, "w", encoding="utf-8") as f:
    f.write(f"写入时间为：{date_str}")
    f.write("\n")
    f.write("Device: CPU \n")
    f.write(f"Interface Time: {(end_time - start_time) * 1000: .2f} ms")
    f.write("\n")
    f.write(f"Memory Usage: {end_memory - start_memory: .2f} MB")
print(f"{file_name} 已创建，并记录baseline模型： {baseLine_version} 数据")