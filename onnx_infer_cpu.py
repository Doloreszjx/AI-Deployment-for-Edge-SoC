import time
import onnxruntime as ort
import numpy as np
from datetime import datetime
import os

# 创建session使用CPU Execution Providers
session = ort.InferenceSession(
    path_or_bytes ="yolov8n.onnx",
    providers=["CPUExecutionProvider"]
)

session_infos = session.get_inputs()[0]
input_name = session_infos.name
input_shape = session_infos.shape
print(session_infos)

# YOLOv8 输入：1x3x640x640
dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)

# 预热
for _ in range(5):
    session.run(None, {input_name: dummy_input})

# 正式测试
start_time = time.time()
outputs = session.run(None, {input_name: dummy_input})
end_time = time.time()

print(f"ONNX Runtime Inference Time: {(end_time - start_time)*1000:.2f} ms")

data_cache_dir = 'data_cache'
os.makedirs(data_cache_dir, exist_ok=True)

file_name = os.path.join(data_cache_dir, 'baseline_ONNX_infos.txt')
date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

with open(file_name, 'a', encoding='utf-8') as f:
    f.write(f'ONNX Info -- 创建时间为{date_str} \n')
    f.write('Device: CPU \n')
    f.write(f'Inference Time: {(end_time - start_time)*1000:.2f} ms')