import os
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import  Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

# os.environ['KERAS_BACKEND'] = 'TENSORFLOW'
(x_train, y_train), (x_test, y_test) =mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1)/255
x_test = x_test.reshape(-1, 28, 28, 1)/255
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

model = Sequential()
#添加第一层卷积
model.add(Convolution2D(
    batch_input_shape=(None, 28, 28, 1),
    filters=36,
    kernel_size=5,
    strides=1,
    padding='same',
    activation='relu'
))

model.add(MaxPooling2D(pool_size=2))

model.add(Convolution2D(
    filters=36,
    kernel_size=5,
    strides=1,
    padding='same',
    activation='relu'
))

model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, verbose=2, batch_size=64)
loss, accuracy = model.evaluate(x_test, y_test)
print("loss is {}, accuracy is {}".format(loss, accuracy))
