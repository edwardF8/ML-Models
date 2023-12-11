from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preq', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(url, names=names)
array = dataframe.values
#separate array into input and output componenets
x = array[:,0:8]
Y= array[:,8]
scaler = StandardScaler().fit(x) #automaticly removes the mean and scales to unti variance
rescaledX = scaler.transform(x)
# fit and multiple transform
#combine fit and transform
numpy.set_printoptions(precision=3)
print(rescaledX[0:5,:])