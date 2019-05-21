
'''
Here are the activation functions

Methods:
    * sigmoid
        sigmoid function
    * relu
        i am sure you know what it is
    * softmax
        usually used for the last layer

Author: Stephen Lee
Email: 245885195@qq.com
Date: 2019.5.21
'''

from numpy import exp,where,sum

def sigmoid(x):
    return 1 / (1 + exp(-1 * x))

def relu(x):
    return where(x > 0, x, 0)

def softmax(x):
    return exp(x) / sum(exp(x))

SUPPORT_ACTIVATION =  {'sigmoid':sigmoid,
                       'relu':relu,
                       'softmax':softmax}
