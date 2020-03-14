from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, InputLayer, Dense, Reshape, Flatten, Dropout, Concatenate, Average, Multiply, Add, add
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, RepeatVector
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import tensorflow as tf

import matplotlib.pyplot as plt
import scipy.misc

import sys
from glob import glob
import os
import scipy
import random
from keras import backend as K
from keras.layers import Layer, AveragePooling2D, Reshape, Flatten
from keras.initializers import Constant

import numpy as np

class myFlatten(Layer):

	def __init__(self, **kwargs):
		super(myFlatten, self).__init__(**kwargs)

	def build(self, input_shape):
		super(myFlatten, self).build(input_shape) 

	def call(self, x):
		return K.reshape(x, shape=(x.shape[0], np.prod(x.shape[1:])))

	def compute_output_shape(self, input_shape):
		return (input_shape[0], np.prod(input_shape[1:]))

class FixedWeightDense(Layer):

	def __init__(self, **kwargs):
		super(FixedWeightDense, self).__init__(**kwargs)

	def build(self, input_shape):
		assert isinstance(input_shape, list)
		super(FixedWeightDense, self).build(input_shape)  # Be sure to call this at the end

	def call(self, x):
		assert isinstance(x, list)
		a, b = x
		L = []
		for i in range(a.shape[0]):
			L.append(K.squeeze(K.dot(K.expand_dims(a[i], axis=0), b[i]), axis=0))
		return K.stack(L, axis=0)

	def compute_output_shape(self, input_shape):
		assert isinstance(input_shape, list)
		shape_a, shape_b = input_shape
		return (shape_a[0], shape_a[1], shape_b[-1])

class FixedWeightConv2D(Layer):

	def __init__(self, **kwargs):
		super(FixedWeightConv2D, self).__init__(**kwargs)

	def build(self, input_shape):
		assert isinstance(input_shape, list)
		super(FixedWeightConv2D, self).build(input_shape)  # Be sure to call this at the end

	def call(self, x):
		assert isinstance(x, list)
		a, b = x
		L = []
		for i in range(a.shape[0]):
			L.append(K.squeeze(K.conv2d(K.expand_dims(a[i], axis=0), b[i], padding='same'), axis=0))
		return K.stack(L, axis=0)

	def compute_output_shape(self, input_shape):
		assert isinstance(input_shape, list)
		shape_a, shape_b = input_shape
		return (shape_a[0], shape_a[1], shape_a[2], shape_b[-1])

class ConKern_Scale_Detector():
	
	def __init__(self, img_rows=96, img_cols=128, img_channels=3, kernel_sizes=None, num_layers=5, output_layers=[0, 1, 2, 3, 4], batch_size=50, num_classes=10):
		
		self.img_rows=img_rows
		self.img_cols=img_cols
		self.img_channels=img_channels
		self.img_shape = (img_rows, img_cols, img_channels, )
		self.kernel_sizes=kernel_sizes
		self.num_layers=num_layers
		self.output_layers=output_layers
		self.batch_size=batch_size
		self.num_classes=num_classes
		
		if self.kernel_sizes is None:
			self.kernel_sizes = [5] * num_layers
		
		I = Input(shape=self.img_shape, batch_shape=(self.batch_size, self.img_rows, self.img_cols, self.img_channels))
		C = Input(shape=(self.num_classes,), batch_shape=(self.batch_size, self.num_classes))
		frames = 64
		L = None
		X = I
		rows = self.img_rows
		cols = self.img_cols
		for i in range(self.num_layers):
			rows = int(rows/2) + rows % 2
			cols = int(cols/2) + cols % 2
			X = Conv2D(frames, kernel_size=self.kernel_sizes[i], strides=2, activation='relu', padding='same')(X)
			X = BatchNormalization(momentum=0.9)(X)
			if i in self.output_layers:
				conv = Dense(1*1*frames*int(frames/2))(Dense(1)(Dense(self.num_classes)(C)))
				conv = BatchNormalization(gamma_initializer=Constant(1/(1.5*frames)**0.5))(Reshape((1, 1, frames, int(frames/2)))(conv))
				Y = FixedWeightConv2D()([X, conv])
				Y = Activation('relu')(Y)
				conv2 = Dense(1*1*int(frames/2)*1)(Dense(1)(Dense(self.num_classes)(C)))
				conv2 = BatchNormalization(gamma_initializer=Constant(1/(0.5*frames)**0.5))(Reshape((1, 1, int(frames/2), 1))(conv2))
				Y = FixedWeightConv2D()([Y, conv2])
				Y = UpSampling2D(2**(i+1))(Y)
				mt = Model([I, C], Y)
				mt.summary()
				upconv = K.constant(value=2**(i+1)/self.kernel_sizes[i], shape=(self.batch_size,self.kernel_sizes[i],self.kernel_sizes[i],1,1))
				Y = FixedWeightConv2D()([Y, upconv])
				mt = Model([I, C], Y)
				mt.summary()
				if L is None:
					L = Y
				else:
					L = Add()([L, Y])
				frames *= 2
		frames = int(frames/2)
		X = myFlatten()(X)
		mat = Dense(frames*rows*cols)(Dense(1)(Dense(self.num_classes)(C)))
		mat = BatchNormalization(gamma_initializer=Constant(1/(0.5*frames)**0.5))(Reshape((frames*rows*cols, 1))(mat))
		X = FixedWeightDense()([X, mat])
		L2 = Add()([L, K.reshape(RepeatVector(self.img_rows*self.img_cols)(X), (self.batch_size,self.img_rows,self.img_cols,1))])
		print(L)
		self.detector = Model([I, C], L)
		self.detector.summary()
		
if __name__ == '__main__':
	detector = ConKern_Scale_Detector()