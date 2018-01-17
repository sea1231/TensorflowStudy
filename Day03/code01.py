import tensorflow as tf
import os
import cv2
import numpy as np
from random import shuffle

# setting
IMG_HEIGHT = 60 # image resize height
IMG_WIDTH = 60 # image resize width
NUM_CHANNEL = 3
NUM_CLASS = 5 # image classfication
IMAGE_DIR_BASE = "../animal_images"
image_dir_list = os.listdir(IMAGE_DIR_BASE)

# image load function
def load_image(addr):
    img = cv2.imread(addr) # load image
    img = cv2.resize(img,(IMG_HEIGHT, IMG_WIDTH),interpolation=cv2.INTER_CUBIC) # resize image
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    class_index = class_index + 1


# shuffle after zip then unzip
zip_list = list(zip(features, labels))
shuffle(zip_list)
features, labels = zip(*zip_list)


features = np.array(features) # tuple -> np.array
labels = np.array(labels) # tuple -> np.array

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
train_features = features[:int(len(features)*0.8)]
train_labels = labels[:int(len(labels)*0.8)]

test_features = features[int(len(features)*0.8):]
test_labels = labels[int(len(labels)*0.8):]

BATCH_SIZE = 50

# 학습을 위한 이미지 처리
def train_data_iterator():
    batch_idx = 0
    while True:
        idxs = np.arange(0, len(train_features)) # 0~training features갯수만큼 index 생성
        np.random.shuffle(idxs) # 섞기

        shuf_features = train_features[idxs] # 섞은 index와 재배치
        shuf_labels = train_labels[idxs] # 섞은 index와 재배치

        batch_size = BATCH_SIZE
        # batch_size 만큼의 단위로 이미지들을 구별(50개의 이미지가 1묶음)
        for batch_idx in range(0, len(train_features), batch_size):
            images_batch = shuf_features[batch_idx:batch_idx + batch_size] / 255.
            images_batch = images_batch.astype("float32")
            labels_batch = shuf_labels[batch_idx:batch_idx + batch_size]
            yield images_batch, labels_batch


images_batch = tf.placeholder(dtype=tf.float32, shape=[None, IMG_HEIGHT*IMG_WIDTH*NUM_CHANNEL])

x_image = tf.reshape(images_batch, [-1, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNEL])

labels_batch = tf.placeholder(dtype=tf.int32, shape=[None])

W_conv1 = tf.get_variable(name="W_conv1", shape=[5,5,3,32], dtype=tf.float32) # 5x5크기의 3채널 kernel 32개 사
b_conv1 = tf.get_variable(name="b_conv1",shape=[32], dtype=tf.float32)

# 입력인 x_image의 dimension은 [batch_size, image_height, image_width, num_channel]로 4차원이다.
# strides의 각 원소는 각 차원에 대한 stride 값이므로 두번째와 3번째 원소가 이미지의 가로/세로에 해당한다.

h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

W_conv2 = tf.get_variable(name="W_conv2", shape=[5,5,32,64], dtype=tf.float32)
b_conv2 = tf.get_variable(name="b_conv2", shape=[64], dtype=tf.float32)

h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# max pooling을 2번 지나면서 이미지 크기가 1/4로 줄었고,
# 마지막 convolution layer에서 64개의 kernel을 사용했으므로 총 값의 개수는 이렇게 된다.
w1 = tf.get_variable("w1", [IMG_HEIGHT//4*IMG_WIDTH//4*64, 1024])
b1 = tf.get_variable("b1", [1024])

# FC 층에 입력하기 위해서 flatten해 준다.
h_pool2_flat = tf.reshape(h_pool2, [-1, IMG_HEIGHT//4*IMG_WIDTH//4*64])

fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w1) + b1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(fc1, keep_prob)

w2 = tf.get_variable("w2", [1024, NUM_CLASS])
b2 = tf.get_variable("b2", [NUM_CLASS])

y_pred = tf.matmul(h_fc1_drop, w2) + b2
class_prediction = tf.argmax(y_pred, 1, output_type=tf.int32)
correct_prediction = tf.equal(class_prediction, labels_batch)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=labels_batch)
loss_mean = tf.reduce_mean(loss)
train_op = tf.train.AdamOptimizer().minimize(loss_mean)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

iter_ = train_data_iterator()

for step in range(500):
    images_batch_val, labels_batch_val = next(iter_)
    accuracy_, _, loss_val = sess.run([accuracy, train_op, loss_mean],
                                      feed_dict={
                                          images_batch:images_batch_val,
                                          labels_batch:labels_batch_val,
                                          keep_prob:0.5
                                      })
    print('Iteration {}: {}, {}'.format(step, accuracy_, loss_val))


print('Training Finished....')

TEST_BSIZE = 50

for i in range(int(len(test_features)/TEST_BSIZE)):
    images_batch_val = test_features[i*TEST_BSIZE:(i+1)*TEST_BSIZE]/255.
    labels_batch_val = test_labels[i*TEST_BSIZE:(i+1)*TEST_BSIZE]
    loss_val, accuracy_ = sess.run([loss_mean, accuracy], feed_dict={
                        images_batch:images_batch_val,
                        labels_batch:labels_batch_val,
                        keep_prob: 1.0
                        })
    print('ACC = {}, LOSS = {}'.format(accuracy_, loss_val))
