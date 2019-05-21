

'''
-------------------------------------------------------------------------
                    cnnframe - a cnn framework
-------------------------------------------------------------------------
This cnn framework is built based on numpy.

Here to use the Model class build a CNN or BP model.

Model(class)
    methods:
    - new Model()       : to build a Model instance
    - add_layer(obj,**) : to add layers the way like Keras
    - summary()         : to check the summary of constructed model
    - train(**)         : to train the model
    - plot_loss()       : to check the changeing of loss

    check cnnframe/cnnframe/test/test_model.py for example

Others

Author: Stephen Lee
Email: 245885195@qq.com
Date: 2019.5.21
-------------------------------------------------------------------------
'''

import numpy as np
import matplotlib.pyplot as plt
from cnnframe.layers.conv2d import Conv2D,PoolingLayer,Flatten
from cnnframe.layers.dense import Dense

DEFAULT_ACCURACY = 0.001

class Model():
    '''
    Common Neural Network model
    '''

    def __init__(self):
        self.layers = []
        self.layers_type = []
        self.train_mse = []
        self.fig_loss = plt.figure()
        self.ax_loss = self.fig_loss.add_subplot(1, 1, 1)

        self.accuracy = 0.1

    def add_layer(self, layer):
        self.layers.append(layer)
        self.layers_type.append(layer.layer_type)
        self.layers_input_shape = []

    def build(self,photo_shape=None):
        '''
        initializer layers
        if only MLP, photo_shape None
        :param photo_shape:
        :return:
        '''
        if 'conv2d' not in self.layers_type and 'flatten' not in self.layers_type:
            # only MLP
            self.model_type = 'MLP'

            for i, layer in enumerate(self.layers[:]):
                if i < 1:
                    layer.is_input_layer = True
                else:
                    layer.initializer(self.layers[i - 1].units)
        else:
            self.model_type = 'CNN'
            for i, layer in enumerate(self.layers[:]):
                self.layers_input_shape.append(photo_shape)

                photo_shape = layer.initializer(photo_shape)


    def summary(self):
        for i, layer in enumerate(self.layers[:]):
            print('----------------------------------------')
            print('                layer %d          ' % i)
            if layer.layer_type == 'conv2d':
                print('conv2d layer')
                print('kernel       ', layer.kernel)
                print('stride       ', layer.stride)
                print('weight.shape ', np.shape(layer.weight))
                print('bias.shape   ', np.shape(layer.bias))
                print('input shape  ', layer.input_shape)
                print('activation   ', layer.activation)

            if layer.layer_type == 'pooling':
                print('pooling size ',layer.kernel_size )
                print('stride       ',layer.stride)
                print('poolint type  ',layer.pooling_type)

            if layer.layer_type == 'flatten':
                print('flatten layer')
                print('units        ',layer.units)
                print('input shape  ',layer.input_shape)

            if layer.layer_type == 'dense':
                print('dense layer')
                print('units        ', layer.units)
                print('weight.shape ', np.shape(layer.weight))
                print('bias.shape   ', np.shape(layer.bias))

            print('----------------------------------------')

    def train(self, xdata, ydata, train_round, accuracy=None,plot_loss=True):
        '''model train procedure'''

        self.train_round = train_round
        accuracy = accuracy if accuracy else DEFAULT_ACCURACY
        self.accuracy = accuracy

        if(plot_loss):
            self.ax_loss.hlines(self.accuracy, xmin=0, xmax=self.train_round * 1.1,
                                    colors='r', label='accuracy')

        x_shape = np.shape(xdata)
        for round_i in range(train_round):
            all_loss = 0
            for row in range(x_shape[0]):
                _xdata = xdata[row]
                _ydata = ydata[row]

                # forward propagation
                for i,layer in enumerate(self.layers):
                    _xdata = layer.forward_propagation(_xdata)

                # calculate loss and gradient
                loss, gradient = self.cal_loss(_ydata, _xdata)
                all_loss = all_loss + loss

                # back propagation
                # the input_layer does not upgrade
                for layer in self.layers[::-1]:
                    gradient = layer.back_propagation(gradient)

            mse = all_loss / x_shape[0]
            self.train_mse.append(mse)

            if(plot_loss):
                self.plot_loss()

            if mse < self.accuracy:
                print('---- reach accuracy----')
                return mse


    def cal_loss(self, ydata, ydata_):
        self.loss = np.sum(np.power((ydata - ydata_), 2))
        self.loss_gradient = 2 * (ydata_ - ydata)
        # vector (shape is the same as _ydata.shape)
        return self.loss, self.loss_gradient

    def plot_loss(self):
        import matplotlib.pyplot as plt

        if self.ax_loss.lines:
            self.ax_loss.lines.remove(self.ax_loss.lines[0])

        self.ax_loss.plot(self.train_mse, linestyle='-', color='#2E68AA')

        if(self.fig_loss is None):
            self.fig_loss = plt.figure()
            plt.ylabel("Loss")
            plt.xlabel("Step")

        plt.ion()
        plt.show()
        plt.pause(0.1)



