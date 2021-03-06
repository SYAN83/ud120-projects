#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from sklearn.svm import SVC
svc = SVC(kernel = 'linear')

### slice the training dataset down to 1% 
features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 

t0 = time()
svc.fit(features_train, labels_train)
str1 = "training time: " + str(round(time()-t0, 3)) + "s"
print str1

t1 = time()
labels_pred = svc.predict(features_test)
str2 =  "predicting time: " + str(round(time()-t1, 3)) + "s"
print str2

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_pred, labels_test)
print accuracy

#########################################################

out = open('output_1.txt', 'w')

out.write('sklearn.svm.SVC(kernel = \'linear\')\n')
out.write(str1 + '\n')
out.write(str2 + '\n')
out.write('Accuracy: ' + str(accuracy))

out.close()
