import onnx

# 读取 ONNX 模型
onnx_model = onnx.load('D:/cups_detection_project/yolov8/code/runs/detect/train/weights/best.onnx')

# 检查模型格式是否正确
onnx.checker.check_model(onnx_model)

print('无报错，onnx模型载入成功')

# 以可读的形式打印计算图
print(onnx.helper.printable_graph(onnx_model.graph))
