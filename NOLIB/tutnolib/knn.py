#You take a sample of it nearest neighbors, whichever class is most prevelent is most relevent

#Euclidan distance
# sqrt((deltax) ^2 + (deltay) ^2)
import numpy
import matplotlib.pyplot as plt


def euclideanDistance(x1, x2):
    return(numpy.sqrt(numpy.sum((x1-x2)^2)))

    
class KNN:
    def __init__(self, k = 3):
        self.k = k
    def fit(X, y,): #returns x
        self.X_train = X
        self.Y_train = y
    def predict(self, X,): #returns y
        predicted_labels = [self._predict(x) for x in X]
        return numpy.array(predicted_labels)

    def  _predict(self, x):
        #compute distnaces
        distances = [euclideanDistance(x, X_train) for x_train in self.X_train]
        #find k nearest samples, labels

        # majority vote, most common class label

