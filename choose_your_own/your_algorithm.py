#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()

print 'Training set: ' + str(len(features_train))
print 'Testing set:  ' + str(len(features_test)) + '\n'

### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them

grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
#################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

import datetime
date_time = datetime.datetime.now()
from time import time

### Method

#from sklearn.ensemble import AdaBoostClassifier
#clf = AdaBoostClassifier(n_estimators=150)

#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(n_estimators=3)

#from sklearn import svm
#clf = svm.SVC()

#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

from sklearn import neighbors
clf = neighbors.KNeighborsClassifier(n_neighbors = 7)

### Training
print 'start training...'
t1 = time()
clf.fit(features_train, labels_train)
str1 = "Training time:   "   + str(round(time()-t1, 3)) + "s"
print str1 + '\n'

### Predicting
print 'start predicting...'
t2 = time()
labels_pred = clf.predict(features_test)
str2 = "predicting time: "   + str(round(time()-t2, 3)) + "s"
print str2 + '\n'

### Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, labels_pred)
print 'accuracy: ' + str(accuracy)

### Write to Log

file = open('log.txt', 'a')
file.write('******************************************************\n\n') 
file.write('DateTime: ' + date_time.strftime("%m/%d/%y %H:%M:%S") + '\n\n')
file.write('Method: ' + str(clf) + '\n\n')
file.write(str1 + '\n')
file.write(str2 + '\n')
file.write('Accuracy: ' + str(accuracy) + '\n\n\n')

### Predicted boundary

try:
    prettyPicture(clf, features_test, labels_test, date_time.strftime("%m%d%y_%H%M%S"))
except NameError:
    pass
