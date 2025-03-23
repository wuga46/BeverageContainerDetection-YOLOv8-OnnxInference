import argparse
from ultralytics import YOLO
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=str,default='./runs/detect/train/weights/best.pt')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = get_args()
    model = YOLO(args.model)
    metrics = model.val(data="data.yaml")  # assumes `model` has been loaded
    print('map: ', metrics.box.map)  # mAP50-95
    print('map50: ', metrics.box.map50)  # mAP50
    print('map75: ', metrics.box.map75)  # mAP75
    print('maps: (每个类别的平均精度)', metrics.box.maps)
