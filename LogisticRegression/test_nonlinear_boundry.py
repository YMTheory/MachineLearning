#!/usr/bin/env python
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import linear_logistic as regression

if __name__ == "__main__":

    X,y = regression.loadDataSet('ex2data2.txt')
    poly = PolynomialFeatures(6)
    XX = poly.fit_transform(X[:,1:3])

    #figures, axes = plt.subplots(1,3, sharey = True, figsize=(17,5))
    rate = 0.001
    maxLoop = 10000
    epsilon = 0.01
    results,timeConsumed = regression.logistic_bgd(rate,maxLoop,epsilon,XX,y)
    theta,errors = results
    """
    errorFig = plt.figure()
    title = 'error decrease with iterations'
    ax = errorFig.add_subplot(111,title=title)
    ax.plot(range(len(errors)), errors)
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Cost J')
    plt.show()
    """

    x1Min = X[:, 1].min()
    x1Max = X[:, 1].max()
    x2Min = X[:, 2].min()
    x2Max = X[:, 2].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1Min, x1Max),np.linspace(x2Min, x2Max))
    h = regression.sigmoid1(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(theta))
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, [0.5], colors='b', linewidth=.5)


    score1_fal =  []
    score2_fal =  []
    score1_pass = []
    score2_pass = []

    score1,score2,result = np.loadtxt("ex2data2.txt",delimiter=',',unpack=True)
    for i in range(len(result)):
        if result[i] == 1:
            score1_pass.append(score1[i])
            score2_pass.append(score2[i])
        else:
            score1_fal.append(score1[i])
            score2_fal.append(score2[i])

    plt.plot(score1_pass,score2_pass,'+',color='blue')
    plt.plot(score1_fal,score2_fal,'*',color='red')
    #xx = np.linspace(0,1,100)
    #theta1 = -(theta[0]+theta[1]*xx)/theta[2]
    #plt.plot(xx,theta1,color='black')
    plt.show()
