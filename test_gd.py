#!/usr/bin/env python
# coding=utf-8

import regression
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

if __name__ == "__main__":
    X, y =regression.loadDataSet('ex1data1.txt');

    m,n = X.shape
    theta = np.ones((m+1,1))
    X = np.concatenate((np.ones((1,n)),X),axis=0)

    rate = 0.01
    maxLoop = 1500
    epsilon = 0.01

    result, timeConsumed = regression.bgd(rate,maxLoop,epsilon,X,y)

    theta,errors,thetas = result
    print(timeConsumed)

    #a,b = np.loadtxt("ex1data1.txt",delimiter=',',unpack=True)
    #xx = np.linspace(0,30,500)
    #pred = theta[0]+theta[1]*xx
    #plt.plot(a,b,'+',color='black')
    #plt.plot(xx,pred,color='blue')

    errorFig = plt.figure()
    title = 'error decrease with iterations'
    ax = errorFig.add_subplot(111,title=title)
    ax.plot(range(len(errors)), errors)
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Cost J')

#    regression.J(theta,X,y)


#    size = 100
#    theta0Vals = np.linspace(0,0.4,size)
#    theta1Vals = np.linspace(0,2.2,size)
#    JVals = np.zeros((size,size))
#    for i in range(size):
#        for j in range(size):
#            col = np.array([theta0Vals[i],theta1Vals[j]])
#            JVals[i,j] = regression.J(col,X,y)

#    theta0Vals, theta1Vals = np.meshgrid(theta0Vals, theta1Vals)
 #   JVals = JVals.T
 #   contourSurf = plt.figure()
 #   ax = contourSurf.gca(projection='3d')

 #   ax.plot_surface(theta0Vals, theta1Vals, JVals,  rstride=2, cstride=2, alpha=0.3,
 #               cmap=cm.rainbow, linewidth=0, antialiased=False)
 #   ax.plot(thetas[0], thetas[1], 'rx')
 #   ax.set_xlabel(r'$\theta_0$')
 #   ax.set_ylabel(r'$\theta_1$')
 #   ax.set_zlabel(r'$J(\theta)$')


#    contourFig = plt.figure()
#    ax = contourFig.add_subplot(111)
#    ax.set_xlabel(r'$\theta_0$')
#    ax.set_ylabel(r'$\theta_1$')

#    CS = ax.contour(theta0Vals, theta1Vals, JVals, np.logspace(-2,3,20))
#    plt.clabel(CS, inline=1, fontsize=10)

    # 绘制最优解
#    ax.plot(theta[0,0], theta[1,0], 'rx', markersize=10, linewidth=2)

    # 绘制梯度下降过程
#    ax.plot(thetas[0], thetas[1], 'rx', markersize=3, linewidth=1)
#    ax.plot(thetas[0], thetas[1], 'r-')








    plt.show()
