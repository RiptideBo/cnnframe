'''
Convolution layers
    - Conv2D:
        layer for convolution
        (kernel, stride ...)

    - PoolingLayer:
        layer for pooling

    - Flatten:
        layer for flattening,
        connecting convolution layer and dense layer(common bp layer)

Common:
    all layers implements methods:
        * forward_propagation
        * backward_propagation

Author: Stephen Lee
Email: 245885195@qq.com
Date: 2019.5.21
'''

import numpy as np
from cnnframe.acfuns import *


class Conv2D():
    def __init__(self,kernel,stride,activation=None,
                 padding='same',learning_rate = None):
        '''
        :param kernel: outchannel,width,height,inchannel (width and height should be 奇数)
        :param slide: step_width,step_height,
        '''
        self.kernel = kernel
        self.stride = stride
        self.activation_name = activation
        self.activation = None
        self.padding = padding
        self.layer_type = 'conv2d'

        if learning_rate is None:
            learning_rate = 0.1
        self.learning_rate = learning_rate

        self.kernel_width = self.kernel[1]
        self.kernel_height = self.kernel[2]
        self.kernel_inchannel = self.kernel[3]
        self.kernel_outchannel = self.kernel[0]


    def _cal_shapes(self,input_shape):
        '''
        * calculate the 'valid' feature map size
        * calculate the padding  of input data
        * calculate the padding shape of output feature map

        :param input_shape: width,height,inchannel
        :return:
        '''

        # convolution center locations on input
        self.input_shape = input_shape

        self.conv_locations_w = np.arange(0, input_shape[0], self.stride[0])
        self.conv_locations_h = np.arange(0, input_shape[1], self.stride[1])

        # valid feature map
        self.feature_map_size = np.asarray([len(self.conv_locations_w), len(self.conv_locations_h)])
        self.feature_map_pixel_num = self.feature_map_size[0] * self.feature_map_size[1]

        # padding input
        self.padding_input_left = int(np.floor(self.kernel_width / 2))
        self.padding_input_top = int(np.floor(self.kernel_height / 2))

        differ_input_right = input_shape[0] - 1 - max(self.conv_locations_w)
        if differ_input_right < self.padding_input_left:
            self.padding_input_right = self.padding_input_left - differ_input_right
        else:
            # stride size > kernel/2
            self.padding_input_right = 0

        differ_input_bottom = input_shape[1] - 1 - max(self.conv_locations_h)
        if differ_input_bottom < self.padding_input_top:
            self.padding_input_bottom = self.padding_input_top - differ_input_bottom
        else:
            self.padding_input_bottom = 0

        self.input_shape_padded = self.input_shape + np.asarray([self.padding_input_left+self.padding_input_right,
                                                                 self.padding_input_top+self.padding_input_bottom,
                                                                 0])
        # 初始输入在padding后图像中的位置
        self.location_input_w = [self.padding_input_left, self.padding_input_left + self.input_shape[0]]
        self.location_input_h = [self.padding_input_top, self.padding_input_top + self.input_shape[1]]

        self.conv_locations_padded_w = self.conv_locations_w + self.padding_input_left
        self.conv_locations_padded_h = self.conv_locations_h + self.padding_input_top


        print('padding input - left,right,top,bottom: ', self.padding_input_left, self.padding_input_right
              ,self.padding_input_top,self.padding_input_bottom)

        # -----------------padding feature map---------------------
        if self.padding == 'same':
            self.padding_map_left = int((input_shape[0] - self.feature_map_size[0])/2)
            self.padding_map_right = input_shape[0] - self.padding_map_left- self.feature_map_size[0]

            self.padding_map_top = int((input_shape[1] - self.feature_map_size[1])/2)
            self.padding_map_bottom = input_shape[1] - self.padding_map_top - self.feature_map_size[1]

            self.feature_map_size_padded = self.feature_map_size + np.asarray([self.padding_map_left+self.padding_map_right,
                                                                               self.padding_map_top+self.padding_map_bottom])

            # 原图像的位置，扩充之后
            self.location_map_w = [self.padding_map_left, self.padding_map_left+self.feature_map_size[0]]
            self.location_map_h = [self.padding_map_top, self.padding_map_top+self.feature_map_size[1]]

        else:
            self.padding_map_left = 0
            self.padding_map_right = 0
            self.padding_map_top = 0
            self.padding_map_bottom = 0
            self.feature_map_size_padded = self.feature_map_size

        self.feature_map_shape = (*self.feature_map_size_padded,self.kernel_outchannel)

        # print('valid feature map ',self.feature_map_size)
        # print(self.location_map_w[0] - self.location_map_w[1])
        # print('feature map shape: ',self.feature_map_shape)

    def _pad_input(self,data):
        input_padded = np.pad(data,
                              pad_width=((self.padding_input_left,self.padding_input_right),
                                         (self.padding_input_top,self.padding_input_bottom),
                                         (0,0)),
                              mode='constant')
        return input_padded

    def _pad_feature_map(self,feature_map):
        # width, height
        feature_map_padded = np.pad(feature_map,
                                    pad_width=((self.padding_map_left,self.padding_map_right),
                                               (self.padding_map_top,self.padding_map_bottom)),
                                    mode='constant')
        return feature_map_padded


    def initializer(self,input_shape):

        self._cal_shapes(input_shape)

        # input_shape (width,height,inchannel)
        '''initializing weight, bias and activation'''

        self.weight = np.asarray(np.random.normal(0, 0.3, size=(self.kernel_outchannel,
                                                                self.kernel_width,
                                                                self.kernel_height,
                                                                self.kernel_inchannel,)))

        self.bias = np.asarray(np.random.normal(0, 0.3, size=(self.kernel_outchannel,
                                                              self.feature_map_size[0],
                                                              self.feature_map_size[1])))


        self.activation = SUPPORT_ACTIVATION.get(self.activation_name)

        if self.activation is None:
            print(' - - No support activation named %s'%str(self.activation_name))
            print(' - - Set activation to relu')
            self.activation = SUPPORT_ACTIVATION.get('relu')

        return self.feature_map_shape

    def forward_propagation(self,inputs):
        '''
        width,height,inchannel
        :param inputs:
        :return:
        '''
        #---------------------------convolution---------------------------
        input_data = self._pad_input(inputs)
        # 记录了卷积中心在原图像上的位置
        self.X = np.ndarray(shape=(self.feature_map_pixel_num,self.kernel_width,
                                   self.kernel_height,self.kernel_inchannel))

        count_ = 0
        for i in self.conv_locations_padded_w:
            for j in self.conv_locations_padded_h:
                data_i = input_data[i - self.padding_input_left:i + self.padding_input_left + 1,
                         j - self.padding_input_top:j + self.padding_input_top + 1,
                         :]
                self.X[count_] = data_i
                count_ += 1

        self.feature_maps = np.ndarray(shape=self.feature_map_shape)
        # width,height,outchannel
        for outchannel in range(self.kernel_outchannel):
            # --------convolute----
            wx = np.multiply(self.X,self.weight[outchannel])
            wx = np.sum(wx,axis=(1,2))

            # --------merge infomations of multi channels
            wx = np.sum(wx,axis=(1))

            # --------reshape to feature map size
            wx = np.reshape(wx,newshape=self.feature_map_size)

            wx_plus_b = wx - self.bias[outchannel]

            if self.padding == 'same':
                wx_plus_b = self._pad_feature_map(wx_plus_b)

            self.feature_maps[:,:,outchannel] = wx_plus_b

            # Image.fromarray(wx_plus_b).show()
        self.output = self.activation(self.feature_maps)

        return self.output

    def cal_activation_gradient(self):
        '''calculate the gradient of wx_plus_b in activation function'''
        if self.activation == sigmoid:
            gradient_activation = np.multiply(self.output, (1 - self.output))
            # width * height * outchannel

        elif self.activation == relu:
            gradient_activation = np.where(self.feature_maps > 0, 1, 0)

        # print('gradient activation shape: ',np.shape(gradient_activation))
        return gradient_activation

    def back_propagation(self,gradient):

        # gradient ： width, height, outchannel

        # print('-----------back propagation-------------')
        # print('input gradient shape',gradient.shape)
        self.gradient_activation = self.cal_activation_gradient()

        gradient = np.multiply(gradient,self.gradient_activation)
        # width, height, outchannel

        # 找到没有使用的部分
        if self.padding == 'same':
            # reverse the featur map padding ， find the original photo
            # extract valid part ：
            # shape: [width, height, outchannel)]
            gradient = gradient[
                       self.location_map_w[0]:self.location_map_w[1],
                       self.location_map_h[0]:self.location_map_h[1],:]


        # ----------------upgrade bias----------------------------
        gradient_bias_ = -1
        gradient_bias = gradient * gradient_bias_
        # (width, height, outchannel)]
        gradient_bias = np.transpose(gradient_bias,(2,0,1))
        # print('gradient bias shape ',gradient_bias.shape)
        self.bias = self.bias + gradient_bias * self.learning_rate
        #----------------------------------------------------------


        # -----------------expand gradient for inchannels------------------------
        # outchannel, inchannel, with, height
        gradient_shape = np.shape(gradient)
        # print('gradient shape before inchannel,',gradient_shape)

        gradient_ = np.ndarray(shape=(self.kernel_inchannel,
                                      gradient_shape[0],
                                      gradient_shape[1],
                                      self.kernel_outchannel))
        # if mean, product 1/inchannel
        gradient_merge_inchannel_ = 1
        for inchannel in range(self.kernel_inchannel):
            gradient_[inchannel,:,:,:] = gradient * gradient_merge_inchannel_


        # print('gradient shape after inchannel: ', gradient_.shape)
        gradient = gradient_.transpose((3,0,1,2))
        # print('gradient transpose to :',gradient.shape)
        # out,in,w,h
        #--------------------------------------------------------------------------

        #-----------------------------Upgrade Weight-------------------------------

        # print('X shape : ',self.X.shape)
        # print('weight shape: ',self.weight.shape)
        # nums, kernel_width, kernel_height, inchannel
        for outchannel in range(self.kernel_outchannel):
            for inchannel in range(self.kernel_inchannel):
                reshape_X = (self.X.shape[0], self.X.shape[1]*self.X.shape[2])
                # change X to  （nums，K-width* K-height）
                gradient_weight_ = self.X[:, :, :, inchannel].reshape(reshape_X)

                reshape_gradient = (1, gradient.shape[-2]*gradient.shape[-1])
                #change gradient  (map width, map height) to (1, nums)
                gradient_i = gradient[outchannel,inchannel,:,:].reshape(reshape_gradient)
                gradient_i = np.dot(gradient_i, gradient_weight_)
                gradient_i = gradient_i.reshape(self.kernel_width,self.kernel_height)

                self.weight[outchannel,:,:,inchannel] = self.weight[outchannel,:,:,inchannel] + gradient_i * self.learning_rate

        # ----------------------------------------------------------------------------------

        # -------------------------------计算后续的梯度--------------------------------------
        # print('----------gradient for X------------')
        gradient_shape = gradient.shape
        # print('current gradient shape: ',gradient_shape)
        # calculate gridents of X
        # add all outchannels gradient
        # print('original input shape ',self.input_shape)
        # print('padding input shape ',self.input_shape_padded)

        reshape_size = (gradient_shape[0],gradient_shape[1],
                        gradient_shape[2] * gradient_shape[3],1,1)

        gradient = gradient.reshape(reshape_size)
        gradient = np.tile(gradient, (1,1,1,self.kernel_width,self.kernel_height))

        gradient = gradient.transpose((2,0,1,3,4))
        '''num , outchannel, inchannel, k_width, k_height'''
        # print('gradient ',np.shape(gradient))

        gradient_x_ = self.weight.transpose((0,3,1,2))
        '''outchannel, inchannel, k_width, k_height'''
        # print('gradient x+ ',np.shape(gradient_x_))

        gradient_X = np.multiply(gradient,gradient_x_)
        '''add the outchannel gradient'''
        gradient_X = np.sum(gradient_X,axis=1)

        # print('gradient x_ ',gradient_X.shape)

        # -------------------
        gradient_output = np.zeros((self.kernel_inchannel,
                                    self.input_shape_padded[0],
                                    self.input_shape_padded[1]))
        count_num = 0
        for i in self.conv_locations_padded_w:
            for j in self.conv_locations_padded_h:
                # k_width , k_height
                gradient_output[:,
                i - self.padding_input_left:i + self.padding_input_left + 1,
                j - self.padding_input_top:j + self.padding_input_top + 1] \
                    += gradient_X[count_num]
        # -------------------

        # print('output gradient shape padded: ',gradient_output.shape)

        # ------截取在填充后图像中，原图像大小
        gradient_output = gradient_output[:,
                          self.location_input_w[0]:self.location_input_w[1],
                          self.location_input_h[0]:self.location_input_h[1]]

        self.gradient = np.transpose(gradient_output,(1,2,0))
        # print('output gradient shape finnal' , self.gradient.shape)
        #--------------------------------------------------------------------------------------
        #转化成channel * 1 * nums
        return self.gradient


class PoolingLayer():
    def __init__(self,kernel_size,stride=None,pooling_type='average'):
        '''

        :param kernel_size: width,height
        :param stride:
        '''
        self.kernel_size = kernel_size

        if stride is None:
            stride = (2,2)
        self.stride = stride
        self.layer_type = 'pooling'
        self.pooling_type = pooling_type


    def initializer(self,input_shape):
        # input_shape, width,height,channel

        self.input_shape = input_shape
        self.locations_w = np.arange(0, input_shape[0], self.stride[0])
        self.locations_h = np.arange(0, input_shape[1], self.stride[1])

        self.output_map_shape = (len(self.locations_w),len(self.locations_h),input_shape[2])
        self.output_pixel_num = self.output_map_shape[0] * self.output_map_shape[1]

        differ_w = input_shape[0] - max(self.locations_w) + 1
        if differ_w < self.kernel_size[0]:
            self.padding_input_right = self.kernel_size[0] - differ_w
        else:
            self.padding_input_right = 0

        differ_h = input_shape[1] - max(self.locations_h) + 1
        if differ_h < self.kernel_size[1]:
            self.padding_input_bottom = self.kernel_size[1] - differ_h
        else:
            self.padding_input_bottom = 0


        self.padding_input = ((0,self.padding_input_right),
                              (0,self.padding_input_bottom),
                              (0,0))

        self.input_shape_padded = (input_shape[0] + self.padding_input_right,
                                   input_shape[1] + self.padding_input_bottom,
                                   input_shape[2])


        return self.output_map_shape

    def forward_propagation(self,inputs):

        inputs = np.pad(inputs,pad_width=self.padding_input,mode='edge')


        self.X = np.zeros(shape=(self.output_pixel_num,*self.kernel_size,self.input_shape[2]))

        count = 0
        for i in self.locations_w:
            for j in self.locations_h:
                data_i = inputs[i:i+self.kernel_size[0],
                         j:j+self.kernel_size[1],
                         :]
                self.X[count] = data_i
                count += 1

        if self.pooling_type == 'average':

            self.output = np.mean(self.X,axis=(1,2))
            self.output = self.output.reshape(self.output_map_shape)


            return self.output

    def back_propagation(self,gradient):
        ''

        gradient_shape = np.shape(gradient)

        self.gradient = np.zeros(shape=self.input_shape_padded)

        for i in range(gradient_shape[0]):
            for j in range(gradient_shape[1]):
                location_x = self.locations_w[i]
                location_y = self.locations_h[j]
                self.gradient[location_x:location_x+self.kernel_size[0]+1,
                location_y:location_y+self.kernel_size[1]+1,:] += gradient[i,j,:]/self.kernel_size[0]

        self.gradient = self.gradient[:self.input_shape[0],
                        :self.input_shape[1],
                        :]


        return self.gradient


class Flatten():
    '''flatten the  output feature maps of last con2d'''
    def __init__(self,units=None,activation=None):
        self.units = None
        self.activation = None
        self.layer_type = 'flatten'


    def initializer(self,input_shape):
        '''
        :param map_shape: width,height,channel
        :return:
        '''

        cal_units = input_shape[0] * input_shape[1] * input_shape[2]

        self.units = cal_units
        self.input_shape = input_shape

        return self.units

    def forward_propagation(self,inputs):
        '''
        flatten the feature maps
        :param inputs: feature maps with [width,height,channel]
        :return: np.array
        '''
        self.input_shape = np.shape(inputs)
        self.outputs = inputs.flatten()

        return self.outputs

    def back_propagation(self,gradient):
        '''
        reshape the gradient to the same shape of feature map
        :param gradient:
        :return:
        '''
        if isinstance(gradient,np.matrix):
            gradient = gradient.getA()

        self.gradiet = np.reshape(gradient,newshape=self.input_shape)

        return self.gradiet

