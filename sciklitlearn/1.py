from sklearn import datasets
from sklearn import svm
#preloaded datasets
iris = datasets.load_iris()
digits = datasets.load_digits() 
#SVC, support a vector classifcation, takes a estimator for classification
clf = svm.SVC(gamma=0.001, C=100.)
#classifer estimator needs to be fitted, we sleect the dataset with the -1 which makes an arr
#with every item but th elawst one