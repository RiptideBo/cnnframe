'''
Dense
    common layer for BP network

    implements methods:
        * forward_propagation
        * backward_propagation
    (java is perfect, no doubt)

Author: Stephen Lee
Email: 245885195@qq.com
Date: 2019.5.21
'''


import numpy as np
from cnnframe.acfuns import *

class Dense():
    '''
    Layers of BP neural network
    '''

    support_activation = {'sigmoid': sigmoid,
                          'relu': relu}

    def __init__(self, units, activation=None, learning_rate=None, is_input_layer=False):
        '''
        common connected layer of bp network
        :param units: numbers of neural units
        :param activation: activation function
        :param learning_rate: learning rate for paras
        :param is_input_layer: whether it is input layer or not
        '''
        self.units = units
        self.weight = None
        self.bias = None
        self.activation_name = activation
        self.activation = None
        self.layer_type = 'dense'

        self.input_shape = self.units

        if learning_rate is None:
            learning_rate = 0.3
        self.learn_rate = learning_rate

        self.is_input_layer = is_input_layer

    def initializer(self, back_units):
        '''initializing weight, bias and activation'''

        self.weight = np.asmatrix(np.random.normal(0, 0.3, (self.units, back_units)))
        self.bias = np.asmatrix(np.random.normal(0, 0.3, self.units))

        if self.activation_name is not None:
            if self.activation_name in SUPPORT_ACTIVATION.keys():
                self.activation = SUPPORT_ACTIVATION.get(self.activation_name)
            else:
                print(' - - No support activation named %s' % str(self.activation_name))
                print(' - - Set activation to sigmoid')
                self.activation = SUPPORT_ACTIVATION.get('sigmoid')

        return self.units

    def cal_gradient(self):
        '''calculate the gradient of wx_plus_b in activation function'''
        if self.activation == sigmoid:
            gradient_mat = np.multiply(self.output, (1 - self.output))
            gradient_activation = np.diag(np.ravel(gradient_mat))

        elif self.activation == relu:
            gradient_mat = np.where(self.wx_plus_b > 0, 1, 0)
            gradient_activation = np.diag(np.ravel(gradient_mat))

        elif self.activation == softmax:
            gradient_mat_2 = np.tile(self.output,(self.units,1))
            gradient_mat_1 = gradient_mat_2.T
            gradient_activation = gradient_mat_1 - np.multiply(gradient_mat_1,gradient_mat_2)
        else:
            gradient_activation = 1


        return gradient_activation

    def forward_propagation(self, xdata):
        '''calculate output'''

        self.xdata = xdata

        if self.is_input_layer:
            # input layer
            self.wx_plus_b = xdata
            self.output = xdata
            return xdata
        else:

            self.wx_plus_b = np.dot(self.xdata, self.weight.T) - self.bias

            if self.activation is not None:
                self.output = self.activation(self.wx_plus_b)
            else:
                self.output = self.wx_plus_b

                print(self.output)
            return self.output



    def back_propagation(self, gradient):
        '''back_proapgation:calculate gradient and upgrade paras'''

        gradient_activation = self.cal_gradient()  # i * i 维
        gradient = np.asmatrix(np.dot(gradient, gradient_activation))

        self._gradient_weight = np.asmatrix(self.xdata)
        self._gradient_bias = -1
        self._gradient_x = self.weight
        self.gradient_weight = np.dot(gradient.T, self._gradient_weight)
        self.gradient_bias = gradient * self._gradient_bias
        self.gradient = np.dot(gradient, self._gradient_x)
        # ----------------------upgrade
        # -----------the Negative gradient direction --------
        self.weight = self.weight - self.learn_rate * self.gradient_weight
        self.bias = self.bias - self.learn_rate * self.gradient_bias

        return self.gradient

