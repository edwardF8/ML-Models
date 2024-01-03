import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split

#analyze data
winedataset = pd.read_csv('data/winequality-white.csv', sep=";",names=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"])
winedataset.dropna(inplace=True)


train,test = train_test_split(winedataset, test_size=0.2)


x_train = train[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]]
y_train = train['quality']


x_test = test[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]]
y_test = test['quality']


#Neural Network implemntation
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=16,activation='sigmoid'),
    tf.keras.layers.Dense(units=8,activation='sigmoid'),
    tf.keras.layers.Dense(units=1,activation='relu')
])

#training

#compile using binarycrosstropy loss function
from tensorflow.keras.losses import BinaryCrosstropy
model.complie(loss=BinaryCrosstropy())

#fit the model using our data
print("Training Model on trainning data:")
model.fit(x_train,y_train,epochs=100)
#pass validation data in before badding validation_data=x,y in the .fit

print("\nEvaluating Model Preformance")
predictions = model.evaluate(x_test,x_train,batch_size = 128)
print("Test Loss, Test Acc: ", predictions)







# The baseline performance for RMSE around 0.148
