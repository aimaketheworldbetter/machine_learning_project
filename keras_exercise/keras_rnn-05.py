#利用RNN实现图片识别
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN
from keras.optimizers import Adam

(x_train, y_train), (x_test, y_test) =mnist.load_data()
# print(x_train.shape, y_train.shape)

x_train = x_train.reshape(-1, 28, 28)/255
x_test = x_test.reshape(-1, 28, 28)/255

# print(x_train.shape, y_train.shape)
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

time_steps = 28
input_size = 28
batch_size = 50
batch_index = 0
output_size = 10
lr = 0.001

model = Sequential()
model.add(SimpleRNN(
    batch_input_shape=(None, time_steps, input_size),
    units = 50,
    unroll=True
))

model.add(Dense(output_size))
model.add(Activation('softmax'))

model.compile(optimizer=Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, verbose=2, batch_size=100)

loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print("loss is {}, accuracy is {}".format(loss, accuracy))
print(model.summary())