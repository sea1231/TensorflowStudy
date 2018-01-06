import numpy as np
import os
import cv2
from random import shuffle

# setting
IMG_HEIGHT = 60 # image resize height
IMG_WIDTH = 60 # image resize width
NUM_CHANNEL = 3
NUM_CLASS = 5 # image classfication
IMAGE_DIR_BASE = "../animal_images"
image_dir_list = os.listdir(IMAGE_DIR_BASE)
print(image_dir_list)


# image load function
def load_image(addr):
    img = cv2.imread(addr) # load image
    img = cv2.resize(img,(IMG_HEIGHT, IMG_WIDTH),interpolation=cv2.INTER_CUBIC) # resize image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) # type change
    return img

# image load
class_index = 0
features = []
labels = []

for dir_name in image_dir_list:
    image_file_list = os.listdir(IMAGE_DIR_BASE + os.sep + dir_name) # image folder
    for file_name in image_file_list:
        image = load_image(IMAGE_DIR_BASE + os.sep + dir_name + os.sep + file_name)
        features.append(image.ravel()) # image 를 1차원 벡터로 변환
        labels.append(class_index)
    class_index += class_index

# shuffle after zip then unzip
zip_list = list(zip(features, labels))
shuffle(zip_list)
features, labels = zip(*zip_list)

print(type(features))

features = np.array(features) # tuple -> np.array
labels = np.array(labels) # tuple -> np.array









