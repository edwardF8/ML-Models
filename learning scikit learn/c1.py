# plotting functions
import matplotlib.pyplot as plt
x = [i for i in range(10)]
y = [2* i for i in range (10)]
plt.plot(x, y) #plots y vs x, linear line
#plt.scatter(x, y) plots points
plt.xlabel('x axis')
plt.ylabel('y axis')

#plt.show()


#saving a model

from sklearn.externals import joblib
#after training model, you can save a model as a .sav
#filename = 'model.sav'
#joblib.dump(clf, filename)
#after you want to run the program
# clf = joblib.load(filename), opens whole saved model

#classifcation
