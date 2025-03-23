import cv2
from ultralytics import YOLO

# 加载自定义模型
model_path = r"D:/BeverageContainerDetection-YOLOv8-Onnx_Inference/code/runs/detect/train/weights/best.pt" # 替换为你的模型路径
model = YOLO(model_path)

# 打开摄像头
cap = cv2.VideoCapture(0)  # 0是默认摄像头

while True:
    # 读取摄像头帧
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头帧")
        break

    # 推理
    results = model.predict(frame, conf=0.65)  # conf是置信度阈值

    # 解析结果并绘制检测框
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 获取边界框坐标
            cls_id = int(box.cls[0])  # 类别ID
            conf = box.conf[0]  # 置信度
            label = model.names[cls_id]  # 类别名称

            # 绘制边界框和标签
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 显示实时画面
    cv2.imshow("Hand Detection", frame)

    # 按'q'退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()