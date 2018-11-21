#!/usr/bin/env python
# coding=utf-8

import regression
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

if __name__ == "__main__":


    srcX, y = regression.loadDataSet('yield.txt')
    m,n = srcX.shape
    srcX = np.array(srcX)
    srcX2 = np.array(srcX*srcX)
    #srcX = np.concatenate((np.power(srcX[0, :],2),srcX), axis=0)
    srcX = np.r_[srcX,srcX2]
    theta = np.ones((m+2,1))
    #print(srcX)
    X = regression.normalization(srcX.copy())
    #X = srcX.copy()
    X = np.concatenate((np.ones((1,n)),X),axis=0)

    rate = 0.0001
    maxLoop = 1000000
    epsilon = 0.01

    result, timeConsumed = regression.bgd(rate,maxLoop,epsilon,X,y)
#print(result)

    theta,errors,thetas = result
    print("Time consumed: ",timeConsumed)
    #print(thetas)

    a,b = np.loadtxt("yield.txt",delimiter=',',unpack=True)
    xx = np.linspace(40,110,500)
    pred = theta[0]+theta[1]*(xx-np.mean(srcX[0,:]))/np.std(srcX[0,:])+theta[2]*(xx*xx-np.mean(srcX[1,:]))/np.std(srcX[1,:])
    plt.plot(a,b,'+',color='black')
    plt.plot(xx,pred,color='blue')

    #errorFig = plt.figure()
    #title = 'error decrease with iterations'
    #ax = errorFig.add_subplot(111,title=title)
    #ax.plot(range(len(errors)), errors)
    #ax.set_xlabel('Number of iterations')
    #ax.set_ylabel('Cost J')

    plt.show()
