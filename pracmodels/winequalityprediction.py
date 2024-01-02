import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split

#analyze data
winedataset = pd.read_csv('data\winequality-white.csv', sep=";",names=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"])
winedataset.dropna(inplace=True)


X = winedataset[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]]
Y=  winedataset[['quality']]

x_train,x_test,y_train,y_test= train_test_split(X,Y, test_size=0.3)


#Neural Network implemntation
#Tensorflow Attempt!!!!!

import tensorflow as tf

normalizer = tf.keras.layers.Normalization(axis=-1)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=8,activation='tanh'),
    tf.keras.layers.Dense(units=1,activation='relu')
])

#training

#compile using binarycrosstropy loss function
model.compile(loss=tf.keras.losses.BinaryCrossentropy())
#fit the model using our data
print("Training Model on trainning data:")
model.fit(x_train,y_train,epochs=100, batch_size=16)
#pass validation data in before badding validation_data=x,y in the .fit

print("\nEvaluating Model Preformance")
predictions = model.evaluate(x_test,y_test,batch_size = 128)
print("Test Loss, Test Acc: ", predictions)



# Boosted Decison Tree implementation


import xgboost
import sklearn.model_selection as sklms

model = xgboost.XGBRegressor()
model.fit(x_train, y_train, eval_set=[(x_test, y_test)])
print(model.score(x_test, y_test))








# The baseline performance for RMSE around 0.148
