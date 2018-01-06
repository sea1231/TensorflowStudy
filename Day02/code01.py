
# 이미지를 불러오고, 4등분으로 짜르기
# 이미지 다루기 연습 예제

import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

image_path = "../animal_images/cat/images.jpeg" # image 경로 설정
img = cv2.imread(image_path) # img type은  numpy array 이다 / img 불러오기

print(img.ndim) # 차원 출력, 이때 img는 3차원 (r,g,b)이기 때문
print(img.shape) # 이미지의 크기 출력, 세로 187 가로 270
print(img.dtype) # 이미지 타입 출력 numpy array
print(img) # numpy array로 변환된 이미지 출력
print(img.tolist()) # numpy array 타입인 이미지를 list 타입으로 변환한 다음 출력

cv2.imshow('test',img) # image show
cv2.waitKey(0) # image show, 아무키를 눌릴때까지 이미지 show, waitkey 지정안하면 이미지 출력 안함
cv2.destroyAllWindows() # image show destory

img = cv2.resize(img,(200,200),interpolation=cv2.INTER_CUBIC) # inter_cubic 알고리즘을 이용해서 이미지를 resize(200 x 200)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB. opencv의 이미지는 bgr type으로 저장됨 이를 rgb로 변경

cv2.imwrite('converted.jpg',img)

# plt.imshow(img)
# plt.show()

# 이미지를 4개의 등분으로 나누는 함수
def plot_image(image):
    fig, axes = plt.subplots(4,4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat): # axes를 flat 화 시킨다.
        row = i//4 # i= index, ax = axes[i]
        col = i % 4
        image_frag = image[row*50:(row+1)*50, col*50:(col+1)*50]
        ax.imshow(image_frag)

        xlabel = '{},{}'.format(row, col)
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

plot_image(img)


