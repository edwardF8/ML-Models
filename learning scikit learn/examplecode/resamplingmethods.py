from pandas import read_csv
import pandas
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
#separate array into input and output componenets 
X = array[:,0:8]
Y = array[:,8]
kflod = KFold(n_splits=10, random_state=7, shuffle=True)## n splits 10 !) for validation
model = LogisticRegression(solver='liblinear') #basic LR model
results = cross_val_score(model, X, Y, cv=kflod)   #accuracy of LR model
print("Accuracy: %.3f%% (%.3F%%)" % (results.mean()*100.0, results.std()*100.0))