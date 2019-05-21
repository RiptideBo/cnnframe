# cnnframe
a simple CNN framework built based on numpy.

### How to use
###### check /test/test_model.py

```python
    from cnnframe.cnn_frame import Model
    from cnnframe.layers.conv2d import Conv2D,PoolingLayer,Flatten
    from cnnframe.layers.dense import Dense

    model = Model()

    model.add_layer(Conv2D(kernel=[4, 3, 3, 1], stride=[2, 2], padding='same',activation='relu'))
    model.add_layer(PoolingLayer(kernel_size=(2,2)))
    model.add_layer(Flatten())
    model.add_layer(Dense(30,activation='relu'))
    model.add_layer(Dense(ydata2.shape[1],activation='sigmoid'))

    model.build(xdata2.shape[1:])
    model.summary()
    model.train(xdata2, ydata2, train_round=100,plot_loss=True)
```


### Layers
check the layers in dir: /cnnframe/cnnframe/layers/*  

All layers imlement methods:
- forward_propagation
- backward_propagation

The layers:
- **Conv2D:**
layer for convolution (kernel, stride ...)

- **PoolingLayer:**
layer for pooling

- **Flatten:**
layer for flattening
connecting convolution layer and dense layer(common bp layer)

- **Dense:**
layer for common BP network

### Activation Functions
- **sigmoid:**
sigmoid function
- **relu:**
sure you know what it is
- **softmax:**
usually used for the last layer

or add your function in /cnnframe/acfuns.py.  

oh!  The backward process.  

it means for now it is not quite  extremely easy to expand more functions


---  
Stephen Lee, 245885195@qq.com, 2019,5,21
