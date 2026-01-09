from onnxruntime.quantization import (
    quantize_static,
    QuantType,
    QuantFormat,
    CalibrationDataReader
)
import calib_data

class DataReader(CalibrationDataReader):
    def __init__(self):
        self.data = calib_data.calibration_data_reader()
        self.enum_data = iter(self.data)

    def get_next(self):
        return next(self.enum_data, None)

quantize_static(
    model_input="../yolov8n.onnx",
    model_output="yolov8n_int8.onnx",
    calibration_data_reader=DataReader(),
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8
)

print("INT8 quantized model saved as yolov8n_int8.onnx")


