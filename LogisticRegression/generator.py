#!/usr/bin/env python
# coding=utf-8

import numpy as np

output = open('linear_sample.txt','w')
string = ''
for i in range(20):
    x1 = np.random.rand()
    print(x1)
    x2 = np.random.rand()
    if(x1+x2<1):
        y = 0
    else:
        y = 1
    s1 = '%f' %x1
    s2 = '%f' %x2
    s3 = '%d' %y
    line = s1 + ','+s2+','+s3+'\n'
    output.writelines(line)

output.close()
