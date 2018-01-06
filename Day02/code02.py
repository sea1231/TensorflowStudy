import numpy as np
import os
import cv2

IMG_HEIGHT = 60 # image resize height
IMG_WIDTH = 60 # image resize width
NUM_CHANNEL = 3
NUM_CLASS = 5 # image classfication

def image_load(addr):
    img = cv2.imread(addr)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


