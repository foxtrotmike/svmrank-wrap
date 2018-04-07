# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 14:49:05 2018
Simple implementation of ranking by Pairwise Transformation
@author: afsar
"""

import itertools
import numpy as np
from scipy import stats
import pylab as pl
from sklearn import svm

def pairwiseTransform(X_train,y_train,blocks):
    comb = itertools.combinations(range(X_train.shape[0]), 2)
    k = 0
    Xp, yp, diff = [], [], []
    for (i, j) in comb:
        if y_train[i] == y_train[j] or blocks[i] != blocks[j]:
            # skip if same target or different group
            continue
        Xp.append(X_train[i] - X_train[j])
        diff.append(y_train[i] - y_train[j])
        yp.append(np.sign(diff[-1]))
        # output balanced classes
        if yp[-1] != (-1) ** k:
            yp[-1] *= -1
            Xp[-1] *= -1
            diff[-1] *= -1
        k += 1
    Xp, yp, diff = map(np.asanyarray, (Xp, yp, diff))
    print len(yp)
    return Xp,yp, diff

class linrankSVM(svm.LinearSVC):               
    def fit(self,X, y, blocks = None, sample_weight = None):    
        if blocks is None:
            blocks = np.ones(len(y))             
        Xp,yp, diff = pairwiseTransform(X_train,y_train,blocks)
        super(linrankSVM, self).fit(Xp,yp, sample_weight = np.abs(diff))
        
if __name__=='__main__':

    np.random.seed(0)
    theta = np.deg2rad(60)
    w = np.array([np.sin(theta), np.cos(theta)])
    K = 20
    X = np.random.randn(K, 2)
    y = [0] * K
    for i in range(1, 3):
        X = np.concatenate((X, np.random.randn(K, 2) + i * 4 * w))
        y = np.concatenate((y, [i] * K))
    
    # slightly displace data corresponding to our second partition
    X[::2] -= np.array([3, 7]) 
    blocks = np.array([0, 1] * (X.shape[0] / 2))
    
    # split into train and test set
    from sklearn import cross_validation 
    cv = cross_validation.StratifiedShuffleSplit(y, test_size=.5)
    train, test = iter(cv).next()
    X_train, y_train, b_train = X[train], y[train], blocks[train]
    X_test, y_test, b_test = X[test], y[test], blocks[test]

    clf = linrankSVM(C=0.10)
    clf.fit(X_train, y_train, blocks = blocks[train])
    for i in range(2):
        tau, _ = stats.kendalltau(
            clf.decision_function(X_test[b_test == i]), y_test[b_test == i])
        print('Kendall correlation coefficient for block %s: %.5f' % (i, tau))
