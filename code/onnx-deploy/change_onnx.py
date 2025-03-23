from ultralytics import YOLO
import torch


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

# 加载YOLOv8模型
model = YOLO(r'D:/BeverageContainerDetection-YOLOv8-Onnx_Inference/code/runs/detect/train/weights/best.pt')    #你的模型路径

# 导出ONNX
model.export(
    format='onnx',
    imgsz=640,  # 与YOLOv8训练参数保持一致
    opset=11,    # ONNX算子集版本
    simplify=True,  # 启用模型简化
    dynamic=False,  # 固定输入维度
    name='best.onnx'  # 输出文件名
)