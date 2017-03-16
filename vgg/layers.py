import tensorflow as tf
import numpy as np
import basic_layer as bl
import time

class VGGLayers:
	def __init__(self, images, trainPhase, weights=None):
		self.input = images
		self.trainPhase = trainPhase
		print("build model started")
		start_time = time.time()
		self.initialVGGLayers()
		# self.initialTextBoxLayers()
		# self.initialOutputs()
		print("build model finished: %ds" % (time.time() - start_time))
		self.prob = self.conv10_2

	def initialVGGLayers(self):
		VGG_MEAN = {'R': 123.68, 'G': 116.779, 'B': 103.939}
		with tf.name_scope('preprocess') as scope:
			images = tf.reshape(self.input, [-1, 300, 300, 3])
			images = images * 255.0
			blue, green, red = tf.split(images, 3, 3)
			images = tf.concat([blue - VGG_MEAN['B'], green - VGG_MEAN['G'], red - VGG_MEAN['R']], 3)
			assert images.shape[1:] == [300, 300, 3]
			self.imgs = images
		self.conv1_1 = bl.conv2d(self.imgs, 3, 64, name='conv1_1')#300
		self.conv1_2 = bl.conv2d(self.conv1_1, 64, 64, name='conv1_2')#300
		self.pool1 = bl.maxPool(self.conv1_2, name='pool1')#150
		self.conv2_1 = bl.conv2d(self.pool1, 64, 128, name='conv2_1')#150
		self.conv2_2 = bl.conv2d(self.conv2_1, 128, 128, name='conv2_2')#150
		self.pool2 = bl.maxPool(self.conv2_2, name='pool2')#75
		self.conv3_1 = bl.conv2d(self.pool2, 128, 256, name='conv3_1')#75
		self.conv3_2 = bl.conv2d(self.conv3_1, 256, 256, name='conv3_2')#75
		self.conv3_3 = bl.conv2d(self.conv3_2, 256, 256, name='conv3_3')#75
		self.pool3 = bl.maxPool(self.conv3_3, name='pool3')#38
		self.conv4_1 = bl.conv2d(self.pool3, 256, 512, name='conv4_1')#38
		self.conv4_2 = bl.conv2d(self.conv4_1, 512, 512, name='conv4_2')#38
		self.conv4_3 = bl.conv2d(self.conv4_2, 512, 512, name='conv4_3')#38
		self.pool4 = bl.maxPool(self.conv4_3, name='pool4')#19
		self.conv5_1 = bl.conv2d(self.pool4, 512, 512, name='conv5_1')#19
		self.conv5_2 = bl.conv2d(self.conv5_1, 512, 512, name='conv5_2')#19
		self.conv5_3 = bl.conv2d(self.conv5_2, 512, 512, name='conv5_3')#19
		self.pool5 = bl.maxPool(self.conv5_3, stride=1, kernel=3, name='pool5')#19
		self.conv6 = bl.conv2d(self.pool5, 512, 1024, name='conv6')#19
		self.conv7 = bl.conv2d(self.conv6, 1024, 1024, kernel=[1,1], name='conv7')#19
		self.conv8_1 = bl.conv2d(self.conv7, 1024, 256, kernel=[1,1], name='conv8_1')#19
		self.conv8_2 = bl.conv2d(self.conv8_1, 256, 512, strides = [2, 2], name='conv8_2')#10
		self.conv9_1 = bl.conv2d(self.conv8_2, 512, 128, kernel=[1,1], name='conv9_1')#10
		self.conv9_2 = bl.conv2d(self.conv9_1, 128, 256, strides=[2,2], name='conv9_2')#5
		self.conv10_1 = bl.conv2d(self.conv9_2, 256, 128, kernel=[1,1], name='conv10_1')#5
		self.conv10_2 = bl.conv2d(self.conv10_1, 128, 256, strides=[2,2], name='conv10_2')#3
		self.pool6 = tf.nn.avg_pool(self.conv10_2, [1, 3, 3, 1], [1, 1, 1, 1], "VALID")#1
	def initialTextBoxLayers(self):
		self.out1 = bl.conv2d(self.conv4_3, 512, 72, kernel=[1,5], name='out1')
		self.out2 = bl.conv2d(self.conv7, 1024, 72, kernel=[1,5], name='out2')
		self.out3 = bl.conv2d(self.conv8_2, 512, 72, kernel=[1,5], name='out3')
		self.out4 = bl.conv2d(self.conv9_2, 256, 72, kernel=[1,5], name='out4')
		self.out5 = bl.conv2d(self.conv10_2, 256, 72, kernel=[1,5], name='out5')
		self.out6 = bl.conv2d(self.conv11, 256, 72, kernel=[1,1], name='out6')
	def initialOutputs(self):
		outputs = [self.out1, self.out2, self.out3, self.out4, self.out5, self.out6]
		outlist = []
		for out in outputs:
			w = out.shape[2]
			h = out.shape[1]
			out_reshaped = tf.reshape(out, [-1, w*h*12, 6])
			outlist.append(out_reshaped)
		formatted_outs = tf.concat(outlist, 1)

		pred_labels = formatted_outs[:, :, 4:]
		pred_locs = formatted_outs[:, :, :4]
		self. outputs = outputs
		self.pred_labels = pred_labels
		self.pred_locs = pred_locs