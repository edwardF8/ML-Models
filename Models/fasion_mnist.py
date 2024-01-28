from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

import matplotlib.pyplot as plt

#Scaling Training Data 
x_train = x_train/255
x_test = x_test/255
x_train.reshape(60000,28,28,1)
x_test.reshape(10000,28,28,1)

#one hot-encoding
from tensorflow.keras.utils import to_categorical
ycat_train = to_categorical(y_train,10)
ycat_test = to_categorical(y_test,10)

#Model setup
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten,Dropout
convModel = Sequential([
    Conv2D(filters=32, kernel_size=(4,4),input_shape=(28,28,1), padding='SAME', activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(filters=64, kernel_size=(2,2), padding='SAME', activation='sigmoid'),
    MaxPool2D(pool_size=(2,2)),
    Flatten(),
    Dense(units = 256, batch_size=16, activation='relu'),
    Dropout(0.2),
    Dense(units=64, batch_size=8, activation='relu'),
    Dense(units=10, activation='softmax')
])
convModel.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
earlyStopCallback = EarlyStopping(monitor='val_loss', patience=2)
convModel.fit(x_train,ycat_train, epochs=100,validation_data=[x_test,ycat_test], callbacks=[earlyStopCallback])

import pandas as pd
metrics = pd.DataFrame(convModel.history.history)
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()
plt.show()