from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, InputLayer, Dense, Reshape, Flatten, Dropout, Concatenate, Average, Multiply, Maximum
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, RepeatVector, Lambda
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
import random, keras
from keras import backend as K
from keras.layers import Layer, AveragePooling2D, Reshape, Flatten
from keras.initializers import Constant, Ones

import numpy as np

class ChannelProjector2D(Layer):
	
	def __init__(self, out_channels, kernel_initializer='glorot_uniform', b_initializer='zeros', **kwargs):
		super(ChannelProjector2D, self).__init__(**kwargs)
		self.out_channels = out_channels
		self.kernel_initializer = keras.initializers.get(kernel_initializer)
		self.b_initializer = keras.initializers.get(b_initializer)
		
	def build(self, input_shape):
		super(ChannelProjector2D, self).build(input_shape)
		self.W = self.add_weight(shape=[input_shape[-1], self.out_channels], initializer=self.kernel_initializer, name='W')
		self.b = self.add_weight(shape=[1, self.out_channels], initializer=self.kernel_initializer, name='W')
		
	def call(self, x):
		L_ = tf.unstack(x, axis=1)
		L = []
		for l in L_:
			L.append([K.expand_dims(K.expand_dims(K.dot(t, self.W), axis=1), axis=1) for t in tf.unstack(l, axis=1)])
		frames = []
		return tf.concat([tf.concat(l, axis=2) for l in L], axis=1)
	
	def compute_output_shape(self, input_shape):
		return input_shape[:-1] + [out_channels]
		


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
		L = ChannelProjector2D(10)(I)
		self.detector = Model(I, L)
		self.detector.summary()
		self.detector.compile(loss=detector_loss, optimizer=Adam(0.0002, 0.5, clipnorm=0.2))
		
	def train(self, epochs=1, n_discriminator_update=1, n_generator_update=2, batch_size=64, save_interval=500):

		imgs = []
		labels = []
		bb_coords = []
		
		imgs = [np.array(scipy.misc.imresize(scipy.misc.imread(img), (self.img_rows, self.img_cols))) for img in imgs]
		imgs = [x/127.5 - 1 for x in imgs]
		I = np.array(imgs).astype(np.float32)
		
		C = np.zeros((len(labels), self.num_classes))
		for i in range(len(labels)):
			C[i, labels[i]-1] = 1
		
		Y = np.zeros((len(bb_coords), self.img_rows, self.img_cols, 1))
		for i in range(len(bb_coords)):
			for bb in bb_coords[i]:
				Y[i, bb[0][0]:bb[1][0], bb[0][1]:bb[1][1], :] = 1
		
		self.detector.fit([I, C], Y, epochs=1, batch_size=self.batch_size, validation_split=0.01)

if __name__ == '__main__':
	detector = ConKern_Scale_Detector()