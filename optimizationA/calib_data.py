import numpy as np

def calibration_data_reader(num_samples=50):
    for _ in range(num_samples):
        yield {
            "images": np.random.rand(1, 3, 640, 640).astype(np.float32)
        }
