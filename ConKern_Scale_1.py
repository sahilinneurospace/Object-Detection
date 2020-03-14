from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, InputLayer, Dense, Reshape, Flatten, Dropout, Concatenate, Average, Multiply, Add
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

def pgan_loss(y_true, y_pred):
	print(y_true, y_pred)
	age_true, race_true, y_true = y_true[:, 1], y_true[:, 2:], y_true[:, 0]
	age_pred, race_pred, y_pred = y_pred[:, 1], y_pred[:, 2:], y_pred[:, 0]
	age_loss = tf.reduce_mean(tf.to_float(tf.less(age_true, 20))*tf.maximum(tf.abs(age_pred-age_true)-5, 0) + tf.to_float(tf.logical_and(tf.greater_equal(age_true,20), tf.less(age_true,55)))*tf.maximum(tf.abs(age_pred-age_true)-10, 0) + tf.to_float(tf.logical_and(tf.greater_equal(age_true,55), tf.less(age_true,85)))*tf.maximum(tf.abs(age_pred-age_true)-20, 0) + tf.to_float(tf.greater_equal(age_true,85))*tf.maximum(tf.abs(age_pred-age_true)-30, 0))
	race_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=race_true, logits=race_pred))
	return -(1-y_true)*K.log(1-y_pred) - y_true*(K.log(y_pred)+y_pred*0.5*(age_loss+race_loss))

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
			L.append(K.squeeze(K.dot(K.expand_dims(a[i], axis=0), b[i], padding='same'), axis=0))
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

class ConKern_Scale():
	
	def __init__(self, img_rows=75, img_cols=100, img_channels=3, range_cover=[5, 10, 20, 40, 80], output_layers=[0, 1, 2, 3, 4], batch_size=50, num_classes=10):
		
		self.img_rows=75
		self.img_cols=100
		self.img_channels=3
		self.img_shape = (img_rows, img_cols, img_channels, )
		self.range_cover=[5, 10, 20, 40, 80]
		self.output_layers=[0, 1, 2, 3, 4]
		self.batch_size=batch_size
		self.num_classes=num_classes
		
		self.kernel_sizes = [self.range_cover[0]] + [int((self.range_cover[i]-self.range_cover[i-1])/2) + 1 for i in range(1, len(self.range_cover))]
		
		I = Input(shape=img_shape, batch_shape=(batch_shape, img_rows, img_cols, img_channels))
		C = Input(shape=(num_classes,), batch_shape=(batch_size, num_classes))
		frames = 64
		L = []
		for i in range(len(self.range_cover)):
			X = Conv2D(frames, kernel_size=self.kernel_sizes[i], strides=2, activation='relu', padding='same')(I)
			X = BatchNormalization(momentum=0.9)(X)
			if i in self.output_layers:
				conv = Dense(1*1*frames*int(frames/2))(Dense(1)(Dense(self.num_classes)(C)))
				conv = BatchNormalization(gamma_initializer=Constant(1/(1.5*frames)**0.5))(Reshape((1, 1, frames, int(frames/2)))(conv))
				Y = FixedWeightConv2D()([X, conv])
				Y = Activation('relu')(Y)
				conv2 = Dense(1*1*int(frames/2)*1)(Dense(1)(Dense(self.num_classes)(C)))
				conv2 = BatchNormalization(gamma_initializer=Constant(1/(0.5*frames)**0.5))(Reshape((1, 1, int(frames/2), 1))(conv2))
				Y = FixedWeightConv2D()([Y, conv2])
				Y = Activation('sigmoid')
				for j in range(i+1):
					Y = UpSampling2D()(Y)
				upconv = Constant(value=2**(i+1)/self.kernel_sizes[i], shape=(self.kernel_sizes[i],self.kernel_sizes[i],1,1))
				Y = FixedWeightConv2D()([Y, upconv])
				L.append(Y)
			frames *= 2
		X = Flatten()(X)
		mat = Dense(frames*1)(Dense(1)(Dense(self.num_classes)(C)))
		mat = BatchNormalization(gamma_initializer=Constant(1/(0.5*frames)**0.5))(Reshape((frames, 1))(mat))
		X = FixedWeightDense()([X, mat])
		X = Activation('sigmoid')
		L.append(Reshape((self.img_rows,self.img_cols, 1))(RepeatVector(self.img_rows*self.img_cols)(X)))
		Y = Concatenate()(L)
		self.detector = model([I, C], L)