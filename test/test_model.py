'''
Here is a test for CNN Model using MNIST data
you will see the loss plot every step
'''

import numpy as np

from cnnframe.cnn_frame import Model
from cnnframe.layers.conv2d import Conv2D,PoolingLayer,Flatten
from cnnframe.layers.dense import Dense

def test_mnist():
    import keras.datasets.mnist as mnist
    from sklearn import preprocessing
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print(np.shape(X_train), np.shape(y_train), type(X_train))

    xdata = X_train[:20,:,:]
    ydata = y_train[:20]

    pro_model = preprocessing.LabelBinarizer()
    pro_model.fit(ydata)
    ydata2 = pro_model.transform(ydata)
    print(ydata2.shape)


    xdata2 = np.ndarray((*xdata.shape,1))
    xdata2[:,:,:,0] = xdata
    for sam in range(xdata2.shape[0]):
        xdata2[sam,:,:,0] = preprocessing.minmax_scale(xdata2[sam,:,:,0],feature_range=(0,1))

    model = Model()

    model.add_layer(Conv2D(kernel=[4, 3, 3, 1], stride=[2, 2], padding='same',activation='relu'))
    model.add_layer(PoolingLayer(kernel_size=(2,2)))
    model.add_layer(Conv2D(kernel=[6, 3, 3, 4], stride=[2, 2], padding='same',activation='sigmoid'))
    model.add_layer(PoolingLayer(kernel_size=(2, 2)))
    model.add_layer(Flatten())
    model.add_layer(Dense(30,activation='relu'))
    model.add_layer(Dense(20,activation='relu'))
    model.add_layer(Dense(ydata2.shape[1],activation='sigmoid'))

    model.build(xdata2.shape[1:])

    model.summary()

    model.train(xdata2, ydata2, train_round=100,plot_loss=True)


if __name__ == '__main__':
    test_mnist()
