# 0
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
# 4

# download and load mnist data from https://www.openml.org/d/554
# for this tutorial, the data have been downloaded already in './scikit_learn_data'
X, Y = fetch_openml('mnist_784', version=1, data_home='./scikit_learn_data', return_X_y=True)

# make the value of pixels from [0, 255] to [0, 1] for further process
X = X / 255.

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=1000)
#1:LogisticRegression


classifier1=LogisticRegression()
classifier1.fit(X_train,Y_train)
predictions=classifier1.predict(X_test)
test_accuracy1=classifier1.score(X_test,Y_test)
train_accuracy1=classifier1.score(X_train,Y_train)

print('LogisticRegression:Training accuracy: %0.2f%%' % (train_accuracy1*100))
print('LogisticRegression:Testing accuracy: %0.2f%%' % (test_accuracy1*100))

BernoulliNB

classifier2 = BernoulliNB()
classifier2.fit(X_train, Y_train)
test_accuracy2 = classifier2.score(X_test, Y_test)
train_accuracy2 = classifier2.score(X_train, Y_train)

print('BernoulliNB:Training accuracy: %0.2f%%' % (train_accuracy2 * 100))
print('BernoulliNB:Testing accuracy: %0.2f%%' % (test_accuracy2 * 100))

classifier4 = LinearSVC()
classifier4.fit(X_train, Y_train)
test_accuracy4 = classifier4.score(X_test, Y_test)
train_accuracy4 = classifier4.score(X_train, Y_train)

print('SVM:Training accuracy: %0.2f%%' % (train_accuracy4 * 100))
print('SVM:Testing accuracy: %0.2f%%' % (test_accuracy4 * 100))

classifier3 = LinearSVC(C=0.8)
classifier3.fit(X_train, Y_train)
test_accuracy3 = classifier3.score(X_test, Y_test)
train_accuracy3 = classifier3.score(X_train, Y_train)

print('SVM2:Training accuracy: %0.2f%%' % (train_accuracy3 * 100))
print('SVM2:Testing accuracy: %0.2f%%' % (test_accuracy3 * 100))
