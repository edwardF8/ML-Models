# Load libraries
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy
#loading dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length','petal-width','class'] #name of columns
data = pd.read_csv(url, names=names)

##pd.plotting.scatter_matrix(data) #scatter matrix
array = data.values # converting data to array
features = array[:,0:4] # features
labels = array[:,4] #labels
#separate array into input and output components
scaler = StandardScaler().fit(features)
rescaledX = scaler.transform(features)

kfold = KFold(n_splits = 10, random_state = 7, shuffle = True)
model = LogisticRegression(solver = 'liblinear')
results = sklearn.model_selection.cross_val_score(model, features, labels, cv=kfold)
print(results.mean() * 100)no