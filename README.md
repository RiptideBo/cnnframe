# cnnframe
A simple CNN framework built based on numpy.

---
## How to use
Check [Here](./test/test_model.py) for example  

Easy to build a CNN or BP model the way like Keras

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

---
## [Layers](./cnnframe/layers)
  
All layers imlement methods:
- forward_propagation
- backward_propagation

#### The layers:
- [**Conv2D**](./cnnframe/layers/conv2d.py): 
layer for convolution (kernel, stride ...)

- [**PoolingLayer**](/cnnframe/layers/conv2d.py): 
layer for pooling

- [**Flatten**](/cnnframe/layers/conv2d.py): layer for flattening, connecting convolution layer and dense layer(common bp layer)

- [**Dense**](./cnnframe/layers/dense.py):
layer for common BP network
---
## [Activation Functions](./cnnframe/acfuns.py)
- **sigmoid:**
sigmoid function
- **relu:**
sure you know what it is
- **softmax:**
usually used for the last layer

Or add your function in /cnnframe/acfuns.py.  

oh!  The backward process.  

It means for now it is not quite  extremely easy to expand more functions.  

Maybe update this some other days.

---

## Other test
#### Convolution effect
How would a input Imgae change after the convolution layers, example [here](./test/test_convlayer.py) (dependency: PIL, numpy)


---  
Stephen Lee, 245885195@qq.com, 2019,5,21
