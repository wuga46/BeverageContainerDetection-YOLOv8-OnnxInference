import argparse
from sre_parse import parse

from ultralytics import YOLO
# 定义一个函数，用于获取命令行参数
def get_args():
    # 创建一个参数解析器
    parser = argparse.ArgumentParser()
    # 添加一个参数，用于指定模型路径，默认值为'./runs/detect/train/weights/best.pt'
    parser.add_argument('--model',type=str,default='input your path')
    # 添加一个参数，用于指定图片路径，默认值为'./test.jpg'
    parser.add_argument('--image',type=str, default='input your path')
    # 解析命令行参数
    args = parser.parse_args()
    # 返回解析后的参数
    return args
if __name__ == '__main__':
    args = get_args()
    model = YOLO(args.model)
    results = model.predict(args.image,iou=0.3)  # assumes `model` has been loaded
    results[0].show()
