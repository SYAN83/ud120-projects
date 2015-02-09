#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 1 (Naive Bayes) mini-project 

    use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
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

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

t0 = time()
gnb.fit(features_train, labels_train)
str1 = "training time: "   + str(round(time()-t0, 3)) + "s"
print str1
t1 = time()
labels_pred = gnb.predict(features_test)
str2 = "predicting time: " + str(round(time()-t1, 3)) + "s"
print str2

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_pred, labels_test)
print accuracy

#########################################################

out = open('output.txt', 'w')

out.write('sklearn.naive_bayes.GaussianNB\n\n')
out.write(str1 + '\n')
out.write(str2 + '\n')
out.write('Accuracy: ' + str(accuracy))

out.close()
