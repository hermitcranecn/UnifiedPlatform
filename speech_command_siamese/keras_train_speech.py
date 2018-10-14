import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

def create_keras_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    #model.add(keras.layers.Dropout(0.5))
    #model.add(keras.layers.Flatten())
    model.add(Dense(10, activation='softmax', name ='predictions'))
    return model

keras_model = create_keras_model()
keras_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train  /= 255
x_test /= 255

keras_model.fit(x_train, y_train, batch_size=32, epochs=1)

score = keras_model.evaluate(x_test, y_test, batch_size=32)

keras.models.save_model(keras_model, "./speech.h5")
