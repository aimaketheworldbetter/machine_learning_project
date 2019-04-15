import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

(x_train, y_train), (x_test, y_test) =mnist.load_data()
# print(x_train.shape, y_train.shape)

x_train = x_train.reshape(x_train.shape[0], -1)/255
x_test = x_test.reshape(x_test.shape[0], -1)/255

print(x_train.shape, y_train.shape)
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

print(x_train.shape, y_train.shape)

model = Sequential([
    Dense(32,input_dim=x_train.shape[1]),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train,verbose=2,epochs=10, batch_size=32)
loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)
print(loss, accuracy)