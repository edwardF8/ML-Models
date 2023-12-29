import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split

#analyze data
winedataset = pd.read_csv('data\winequality-white.csv', sep=";",names=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"])

print(winedataset[:10])





# The baseline performance for RMSE around 0.148
