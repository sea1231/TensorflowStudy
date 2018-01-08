import numpy as np
import os
import cv2
from random import shuffle
import tensorflow as tf


# setting
IMG_HEIGHT = 100  # image resize height
IMG_WIDTH = 100  # image resize width
NUM_CHANNEL = 3
NUM_CLASS = 5  # image classfication
IMAGE_DIR_BASE = "../animal_images"
image_dir_list = os.listdir(IMAGE_DIR_BASE)


# image load function
def load_image(addr):
    img = cv2.imread(addr)  # load image
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC)  # resize image
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)  # type change
    return img


# image load
class_index = 0
features = []
labels = []

for dir_name in image_dir_list:
    image_file_list = os.listdir(IMAGE_DIR_BASE + os.sep + dir_name)  # image folder
    for file_name in image_file_list:
        image = load_image(IMAGE_DIR_BASE + os.sep + dir_name + os.sep + file_name)
        features.append(image.ravel())  # image 를 1차원 벡터로 변환
        labels.append(class_index)
    class_index = class_index + 1

# shuffle after zip then unzip
zip_list = list(zip(features, labels))
shuffle(zip_list)
features, labels = zip(*zip_list)

features = np.array(features)  # tuple -> np.array
labels = np.array(labels)  # tuple -> np.array

# # test 하기
# image = features[0] # 1개의 이미지 선택
#
# # 우리는 하나의 이미지를 1차원 벡터로 바꾸었기 때문에, 다시 이를 3차원으로 변경
# image = image.reshape((IMG_HEIGHT,IMG_WIDTH,NUM_CHANNEL))
# # float type 에서 이미지는 int 이기 때문에 이를 int 형식으로 변경
# image = image.astype(np.uint8)

# cv2.imshow('f0',image)
# cv2.imwrite('code02_1.jpg',image)
# cv2.waitKey(0)

# training data 와 test data는 8:2 비중이 일반적으로 사용된다
train_features = features[:int(len(features) * 0.8)]
train_labels = labels[:int(len(labels) * 0.8)]

test_features = features[int(len(features) * 0.8):]
test_labels = labels[int(len(labels) * 0.8):]

BATCH_SIZE = 50


# 학습을 위한 이미지 처리
def train_data_iterator():
    batch_idx = 0
    while True:
        idxs = np.arange(0, len(train_features))  # 0~training features갯수만큼 index 생성
        np.random.shuffle(idxs)  # 섞기

        shuf_features = train_features[idxs]  # 섞은 index와 재배치
        shuf_labels = train_labels[idxs]  # 섞은 index와 재배치

        batch_size = BATCH_SIZE
        # batch_size 만큼의 단위로 이미지들을 구별(50개의 이미지가 1묶음)
        for batch_idx in range(0, len(train_features), batch_size):
            images_batch = shuf_features[batch_idx:batch_idx + batch_size] / 255.
            images_batch = images_batch.astype("float32")
            labels_batch = shuf_labels[batch_idx:batch_idx + batch_size]
            yield images_batch, labels_batch


# ======== tensorflow graph 설정 시작 ========
# input data set
# 이미지의 행렬은 가로x세로x3
images_batch = tf.placeholder(dtype=tf.float32, shape=[None, IMG_HEIGHT * IMG_WIDTH * NUM_CHANNEL])
labels_batch = tf.placeholder(dtype=tf.int32, shape=[None, ])

# layer 1 설정
# weight and bias set,
# 행렬끼리 곱하기 위해서는 행x열 X 행(이전의 열)x열(배출될 원소 갯수)
# W행렬은 “이전 layer의 노드 개수 * 현재 layer의 노드 개수
w1 = tf.get_variable("w1", [IMG_HEIGHT * IMG_WIDTH * NUM_CHANNEL, 2048])
b1 = tf.get_variable("b1", [2048])

# layer 1의 출력벡터
# activation function=relu을 이용해 출력벡터 계산
fc1 = tf.nn.relu(tf.matmul(images_batch, w1) + b1)

# layer 2 설정
w2 = tf.get_variable("w2", [2048, 1024])
b2 = tf.get_variable("b2", [1024])
# 이전 layer의 출력이 이 layer의 입력이 된다.
fc2 = tf.nn.relu(tf.matmul(fc1, w2) + b2)

# layer 3 설정
w3 = tf.get_variable("w3", [1024, 512])
b3 = tf.get_variable("b3", [512])
fc3 = tf.nn.relu(tf.matmul(fc2, w3) + b3)

# layer 4 설정
w4 = tf.get_variable("w4", [512, 256])
b4 = tf.get_variable("b4", [256])
fc4 = tf.nn.relu(tf.matmul(fc3,w4) + b4)

w5 = tf.get_variable("w5", [256, NUM_CLASS])
b5 = tf.get_variable("b5", [NUM_CLASS])

y_pred = tf.matmul(fc4, w5) + b5

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=labels_batch)
loss_mean = tf.reduce_mean(loss)
train_op = tf.train.AdamOptimizer().minimize(loss)

# 결과값에 soft max 적용
# 그럼 결과 y_pred(5개의 값)이 normalized 됨
y_normalized = tf.nn.softmax(y_pred)

# 그 중 가장 큰 값을 선택 [0,0,0,0,1]==sheep 가 됨
y_pred_labels = tf.cast(tf.argmax(y_normalized, 1), tf.int32)

# one hot encoding 이 되어있지 않음이 문제
correct_prediction = tf.equal(y_pred_labels, labels_batch)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# ======== tensorflow graph 설정 끝 ========
sess = tf.Session()

sess.run(tf.global_variables_initializer())

iter_ = train_data_iterator()  # generator를 호출하면 iterator를 반환한다.

for step in range(500):
    # get a batch of data
    # 50개 단위로 이미지를 받아와 images_atch_val, labels_batch_val에 저장
    images_batch_val, labels_batch_val = next(iter_)

    # 테스트를 위해 여러가지 텐서들을 실행
    _, loss_val, accuracy_val = sess.run([train_op, loss_mean, accuracy], feed_dict={
        images_batch: images_batch_val,
        labels_batch: labels_batch_val})

    print(loss_val, accuracy_val)


print('Training Finished....')

TEST_BSIZE = 50

for i in range(int(len(test_features) / TEST_BSIZE)):
    images_batch_val = test_features[i * TEST_BSIZE:(i + 1) * TEST_BSIZE] / 255.
    labels_batch_val = test_labels[i * TEST_BSIZE:(i + 1) * TEST_BSIZE]

    loss_val, accuracy_val = sess.run([loss_mean, accuracy], feed_dict={
        images_batch: images_batch_val,
        labels_batch: labels_batch_val
    })

    print(loss_val, accuracy_val)