#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#引入所需库
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import Adadelta


#%%
batch_size = 128
num_classes = 10 #分类个数
epochs = 10 #训练轮数

#%%

(X_train, y_train), (X_test, y_test) = mnist.load_data() #加载MNIST数据

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') #转换格式，（样本数量，长，款，1）
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

X_train /= 255 #像素大于介于0~255，统一除以255，把像素值控制在0~1的范围
X_test /= 255

y_train = np_utils.to_categorical(y_train, 10) #生成one-hot编码
y_test = np_utils.to_categorical(y_test, 10) 


#%%

#构建模型

model = Sequential()
#第一层卷积层
model.add(Conv2D(filters = 64, kernel_size = (3,3), activation='relu', input_shape = (28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
#第二层卷几层
model.add(Conv2D(filters = 64, kernel_size = (3,3), activation='relu' ))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
#铺平当前节点
model.add(Flatten())
#生成全连接层
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
#定义损失函数学习率
model.compile(loss = 'categorical_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])

#%%
#开始训练
model.fit(X_train, y_train, batch_size = batch_size, verbose = 1, validation_data = (X_test, y_test))

#计算准确率
score = model.evaluate(X_test, y_test, verbose = 0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


#%%
#保存模型
model.save('mnist_model_check.h5')

#%%
#自己手写一张图片，进行准确率测试
import skimage.io 
import matplotlib.pyplot as plt

img1 = skimage.io.imread('3.jpg', as_grey = True)

skimage.io.imshow(img1)
plt.show()

img1 = np.reshape(img1, (1, 28, 28 ,1)).astype('float32')

proba = model.predict_proba(img1,verbose = 0)
result = model.predict_classes(img1, verbose = 0)

print(proba[0])
print(result[0])

