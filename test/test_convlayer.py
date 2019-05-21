import os
import numpy as np

def test_convfunction():
    '''
    Here you will see change of the picture after common Conv2D layer
    '''

    from PIL import Image
    from cnnframe.layers.conv2d import Conv2D

    file = os.path.join("dog.jpg")
    img = Image.open(file)
    data = np.asarray(img)
    data_shape = data.shape
    print('iuput image shape: ',data_shape)

    kernel = [4,5,5,3] #outchannel, width, height, inchannel
    stride = [6,6]

    layer = Conv2D(kernel,stride,padding='same',activation='relu')
    layer.initializer((data_shape))
    data_conved = layer.forward_propagation(data)

    print("output map shape: ", np.shape(data_conved))

    for i in range(np.shape(data_conved)[-1]):
        each_conved = data_conved[:,:,i]
        print(np.shape(each_conved))
        each_img = Image.fromarray(each_conved)
        each_img.show()

    grident = np.random.normal(0,0.4,(layer.feature_map_size_padded[0],
                                      layer.feature_map_size_padded[1],
                                      layer.kernel_outchannel))
    print('gradient shape: ',np.shape(grident))

    layer.back_propagation(grident)


if  __name__ == '__main__':
    test_convfunction()
