import os
import keras
from numpy import shape
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Input

def Conv_Block(x_input, filters, kernel_size, conv_strides, activation, maxpool_size, maxpool_strides):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=conv_strides, activation=activation)(x)
    x = MaxPool2D(pool_size=maxpool_size, strides=maxpool_strides)(x)
    return x

def AlexNetModel():
    x_input = Input(shape=shape)

    # Convolutional Block 1
    x = Conv_Block(x_input, 96, (11, 11), 4, "ReLu", (3, 3), 2)(x)

    # Convolutional Block 2
    x = Conv_Block(x, 256, (5, 5), 1, "ReLu", (3, 3), 2)(x)
