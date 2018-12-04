#!/usr/bin/env python
# coding=utf-8

from sklearn.metrics.pairwise import rbf_kernel
import numpy as np

def linearKernel():
    #线性核函数
    def calc(X,A):
        return X * A.T
    return calc


def rbfKernel(delta):
    #高丝核
    gamma = 1.0/(2*delta**2)
    def calc(X,A):
        return np.mat(rbf_kernel(X,A,gamma=gamma))
    return calc


def getSmo(X,y,C,tol,maxIter,kernel=linearKernel()):

    m,n = X.shape
    K = kernel(X,X)
    ECache = np.zeros((m,2))

    #权重向量
    def w(alphas,b,supportVectorsIndex,supportVectors):
        return (np.multiply(alphas[supportVectorsIndex],y[supportVectors]).T * supportVectors).T

    #预测误差
    def E(i,alphas,b):
        FXi = float(np.multiply(alphas,y).T * K[:,i]) + b
        E = FXi - float(y[i])
        return E

    def predict(X,alphas,b,supportVectorsIndex,supportVectors):
        Ks = kernel(supportVectors,X)
        predicts = (np.multiply(alphas[supportVectorsIndex],y[supportVectorsIndex]).T * Ks + b).T
        predict = np.sign(predicts)
        return predict

    def select(i,alphas,b):
        #选择alpha
        Ei = E(i,alphas,b)

        #选择违反KKT的作为alpha2
        Ri = y[i] - Ei
    
