import os
import random
import shutil

from tqdm import tqdm

image_root = './images'
label_root = './labels'
train_split = 0.7
images = os.listdir(image_root)
train = random.sample(images,int(len(images)*train_split))
os.makedirs('./datasets/yolo/train/images',exist_ok=True)
os.makedirs('./datasets/yolo/train/labels',exist_ok=True)
os.makedirs('./datasets/yolo/val/images',exist_ok=True)
os.makedirs('./datasets/yolo/val/labels',exist_ok=True)
for image in tqdm(images):
    if image in train:
        shutil.copy(os.path.join(image_root, image),'./datasets/yolo/train/images')
        try:
            shutil.copy(os.path.join(label_root, image[:-4]+'.txt'), './datasets/yolo/train/labels')
        except:
            pass
    else:
        shutil.copy(os.path.join(image_root, image), './datasets/yolo/val/images')
        try:
            shutil.copy(os.path.join(label_root, image[:-4] + '.txt'), './datasets/yolo/val/labels')
        except:
            pass

print("done!!!")
