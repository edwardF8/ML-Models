import pandas as pd
import matplotlib.pyplot as plt #visualization of data
import pandas.plotting 
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preq','plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(url, names=names)
description = data.describe() #shows distrubution
shape = data.shape #shows dimenstions
head = data.head() #shows first few rows
dtypes = data.dtypes #shows datatypes for values USEFULL 
print(head)
print(shape)
print(description)
print(dtypes)


data.scatte
plt.show
