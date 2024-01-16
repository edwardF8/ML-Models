import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#analyze data

var = input("TrainingLaptop or PC: ")
csv = ""
if(var == "Laptop"):
    csv = 'ML-Models\\data\\winequality-white.csv'
elif(var == "PC"):
    csv = 'data\\winequality-white.csv'


winedataset = pd.read_csv(csv, sep=";",names=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"])

winedataset.dropna(inplace=True)



plt.show()

X = winedataset[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]].values
Y=  winedataset[['quality']].values



x_train,x_test,y_train,y_test= train_test_split(X,Y, test_size=0.3)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#Neural Network implemntation
#Tensorflow Attempt!!!!!

import tensorflow as tf

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(x_train)



model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(11,)),
    tf.keras.layers.Dense(units=11,activation='tanh'),
    tf.keras.layers.Dense(units=4,activation='tanh'),
    tf.keras.layers.Dense(units=1,activation='relu')
])

#training

#compile 

#fit the model using our data
print("Training Model on trainning data:")
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.metrics.mean_squared_error,metrics=[tf.keras.metrics.mse])
model.fit(x_train,y_train,epochs=50, batch_size=16, validation_data=[x_test,y_test])

#pass validation data in before badding validation_data=x,y in the .fit

losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()

print("\nEvaluating Model Preformance")
predictions = model.evaluate(x_test,y_test,batch_size = 128)
print("Test Loss, Test Acc: ", predictions)



# Boosted Decison Tree implementation

'''
import xgboost
import sklearn.model_selection as sklms

model = xgboost.XGBRegressor()
model.fit(x_train, y_train, eval_set=[(x_test, y_test)])
print(model.score(x_test, y_test))

'''



# The baseline performance for RMSE around 0.148

