import numpy as np
import tensorflow as tf
import cv2
import os

def miniBatch(LabelVector, DataVector, batch_size):

    num_data = LabelVector.shape[0]

    while True:
        idxs = np.arange(num_data)
        np.random.shuffle(idxs)

        # batch_size 만큼 뽑기

        for start in range(0, num_data-batch_size, batch_size):
            miniLabelVector = LabelVector[idxs[start:start+batch_size]]
            miniDataVector = DataVector[idxs[start:start+batch_size]]
            yield (miniLabelVector, miniDataVector)

def weight_variable(shape):
   initial = tf.truncated_normal(shape, stddev=0.1)
   return tf.Variable(initial)

def bias_variable(shape):
   initial = tf.constant(0.1, shape=shape)
   return tf.Variable(initial)

def conv2d(x, W):
   return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') #padding = "SAME" 입력과 같은 형태로 출력해라.

def max_pool_2x2(x):
   return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



#hog = cv2.HOGDescriptor()

train_images = []
tlabels = []


neg_path = os.getcwd() + "/image/neg/neg_train"
counter = 0
for filename in os.listdir(neg_path):
    image = cv2.imread(neg_path+"/"+filename,0)
    image = cv2.resize(image,(70,134))
    #h_image = hog.compute(image)
    #train_images.append(h_image)
    train_images.append(image)
    tlabels.append(0)
    counter += 1

neg_path = os.getcwd() + "/image/neg/neg_test"
for filename in os.listdir(neg_path):
    image = cv2.imread(neg_path+"/"+filename,0)
    image = cv2.resize(image,(70,134))
    train_images.append(image)
    tlabels.append(0)
    counter += 1


pos_path = os.getcwd() + "/image/pos/pos_train"

for filename in os.listdir(pos_path):
    image = cv2.imread(pos_path+"/"+filename,0)
    #h_image = hog.compute(image)
    #train_images.append(h_image)
    train_images.append(image)
    tlabels.append(1)
    counter += 1

pos_path = os.getcwd() + "/image/pos/pos_test"

for filename in os.listdir(pos_path):
    image = cv2.imread(pos_path+"/"+filename,0)
    #h_image = hog.compute(image)
    #test_images.append(h_image)
    train_images.append(image)
    tlabels.append(1)
    counter += 1

# Image 데이터와 Label 데이터를 numpy 데이터로 수정한다
train_images = np.array(train_images)
#train_images = train_images.reshape(counter, 3780, )
train_images = train_images.reshape(counter, 9380, )


tlabels = np.array(tlabels)     # tlabels = (1,counter)
tlabels = tlabels.reshape(counter,1)

# train Label 데이터를 [1 x 2] 의 행렬로 표현한다
train_labels  = np.array(np.zeros(counter*2).reshape(counter,2))
for num in range(counter):
    train_labels[num][int(tlabels[num][0]) - 1] = 1



#-----------------------------------------------------------------

# test file
test_images = []

test_path = os.getcwd()
filename = input("test file name from current file path : ")

image10 = cv2.imread(pos_path+"/"+filename,0)
test_images.append(image10)
image7 = cv2.resize(image10,(0,0), fx="0.7", fy="0.7")
test_images.append(image7)
image5 = cv2.resize(image10,(0,0), fx="0.5", fy="0.5")
test_images.append(image5)
image3 = cv2.resize(image10,(0,0), fx="0.3", fy="0.3")
test_images.append(image3)

#---------------------croping-------------------------------------
window_w = 70
window_h = 134
test_e_images = []
test_e_images_c = []
for e_image in test_images:
    width, height = e_image.shape
    #-----------------------------------------------------------------
    crop_image_counter = 0
    #-----------------------------------------------------------------
    row = 0
    col = 0
    while row + window_h < height:
        while col + window_w < width:
            crop_image = e_image[row:row + window_h, col:col + window_w]
            test_e_images.append(crop_image)
            crop_image_counter += 1
            col += int(window_w * 0.7)
        crop_image = e_image[row:row + window_h, width - window_w:width]
        test_e_images.append(crop_image)
        crop_image_counter += 1
        row += int(window_h * 0.7)
    col = 0
    while col + window_w < width:
        crop_image = e_image[height - window_h:height, col:col + window_w]
        test_e_images.append(crop_image)
        crop_image_counter += 1
    crop_image = e_image[height - window_h:height, width - window_w:width]
    test_e_images.append(crop_image)
    crop_image_counter += 1

    test_e_images_c.append(crop_image_counter)


#-----------------------------------------------------------------
# Tensorflow 코드
#-----------------------------------------------------------------

#x = tf.placeholder("float32", [None, 3780])
x = tf.placeholder("float32", [None, 9380])
y = tf.placeholder("float32", [None, 2])

#W = tf.Variable(tf.zeros([3780,2]))
W = tf.Variable(tf.zeros([9380,2]))
b = tf.Variable(tf.zeros([2]))

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

#x_image = tf.reshape(x, [-1, 126, 30, 1])
x_image = tf.reshape(x, [-1, 70, 134, 1])


h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#h_pool2 = max_pool_2x2(h_conv2)

#W_fc1 = weight_variable([63*15*64,1024])
W_fc1 = weight_variable([35*67*64,1024])

b_fc1 = bias_variable([1024])

#h_pool2_flat = tf.reshape(h_conv2,[-1, 63*15*64])
h_pool2_flat = tf.reshape(h_conv2,[-1, 35*67*64])


h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))

#train_step = tf.train.AdamOptimizer(3e-3).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

train_iter = miniBatch(train_labels,train_images, 100)

print("Learning Start!")
#배치 함수를 짜면 성능을 올릴수도 있을 듯.. 보통 사용하는 배치함수는 알아서 이미지랑 라벨을 [[image[50]],[label[50]]] 이런식으로 올려줌
for i in range(200):
    (trainingLabel, trainingData) = next(train_iter)
    #if i%5 == 0:
    #    train_accuracy = accuracy.eval(feed_dict ={x: test_images, y: test_labels, keep_prob: 1.0})
    #    print('step', i, 'traing accruacy', train_accuracy)
    train_step.run(feed_dict ={x:trainingData, y: trainingLabel, keep_prob: 0.5})

# 전부 학습이 끝나면 테스트 데이터를 넣어 정확도를 계산한다
#test_accuracy = accuracy.eval(feed_dict={x: test_images, y: test_labels, keep_prob: 1.0})
#print('test accuracy', test_accuracy)

