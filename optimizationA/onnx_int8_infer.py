import onnxruntime as ort
import numpy as np
import time
import os
from datetime import datetime

sess_options = ort.SessionOptions()
sess_options.enable_profiling = True

session = ort.InferenceSession(
    "yolov8n_int8.onnx",
    sess_options,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)

# warm-up
for _ in range(5):
    session.run(None, {input_name: dummy_input})

# benchmark
start = time.time()
session.run(None, {input_name: dummy_input})
end = time.time()

print(f"INT8 Inference Time: {(end - start)*1000:.2f} ms")
data_cache_dir = 'data_cache'
os.makedirs(data_cache_dir, exist_ok=True)

file_name = os.path.join('../', data_cache_dir, 'INT8_interface_info.txt')
date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
with open (file_name, 'w', encoding='utf-8') as f:
    f.write(f'INT8 Info -- 创建时间为{date_str} \n')
    f.write('Device: CPU \n')
    f.write(f"INT8 Inference Time: {(end - start)*1000:.2f} ms")

profile_file = session.end_profiling()
print("INT8 profile saved to:", profile_file)
