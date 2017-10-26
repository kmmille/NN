#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import serializers
from chainer import initializers
import chainer.functions as F
import chainer.links as L
import math
import os
import numpy as np

class CNN(chainer.Chain):
	def __init__(self, n_out):
		self.dtype = np.float32
		w = 1/np.sqrt(2)
		initW = initializers.HeNormal(scale=w)
		initbias = initializers.Zero()

		super(CNN,self).__init__(
			conv1 = L.ConvolutionND(3, in_channels=1, out_channels=32, ksize=4, stride=2, initialW=initW, initial_bias=initbias),
			bn1=L.BatchNormalization(n_out,dtype=self.dtype),
			conv2 = L.ConvolutionND(3, in_channels=32, out_channels=32, ksize=3, stride=2, initialW=initW, initial_bias=initbias),
			bn2=L.BatchNormalization(n_out,dtype=self.dtype),
			conv3 = L.ConvolutionND(3, in_channels=32, out_channels=n_out, ksize=2, stride=1, initialW=initW, initial_bias=initbias),
			bn3=L.BatchNormalization(n_out,dtype=self.dtype),
			conv4 = L.ConvolutionND(3, in_channels=32, out_channels=n_out, ksize=6, stride=1, initialW=initW, initial_bias=initbias),
			bn4=L.BatchNormalization(n_out,dtype=self.dtype),
			)

	def __call__(self, x, train):
		with chainer.using_config('train',train):
			h = F.relu(self.bn1(self.conv1(x)))
			h = F.relu(self.bn2(self.conv2(h)))
			h = F.relu(self.bn3(self.conv3(h)))
			h = F.relu(self.bn4(self.conv4(h)))
			return h

class SplitCNN(chainer.Chain):
	def __init__(self, nfeats, out_per_feat):
		self.dtype = np.float32
		w = 1./np.sqrt(2)
		initW = initializers.HeNormal(scale=w)
		initbias = initializers.Zero()

		super(SplitCNN,self).__init__()
		self.cnns = []
		for i in range(nfeats):
			self.forward.append('CNN%s'%i)
			self.add_link('CNN%s'%i, CNN(out_per_feat))
		self.add_link('fc1',F.Linear(out_per_feat*len(inputs),out_size=64,initialW=initW, initial_bias=initbias))
		self.add_link('fc2',F.Linear(64,out_size=2,initialW=initW, initial_bias=initbias))

	def __call__(self,x,t,train):
		inputs = F.split_axis(x,indices_or_sections=nfeats,axis=1)
		outputs = []
		for i,inp in enumerate(inputs):
			outputs.append(getattr(self,self.cnns[i])(inp,train))
		
		h = F.concat(tuple(outputs),axis=1)
		h = self.fc1(h)
		h = self.fc2(h)

		if train:
			self.loss = F.softmax_cross_entropy(x, t)
			self.accuracy = F.accuracy(x, t)
			return self.loss
		else:
			return x

model = SplitCNN(3,32)




