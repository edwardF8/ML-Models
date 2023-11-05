from sklearn import datasets
import numpy as np


#loads example dataset
iris = datasets.load_iris()

#splits features and labels
X = iris.data
y = iris.target

