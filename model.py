import tensorflow as tf
import numpy as np
# import vgg.tb_base as vgg16
import vgg.layers as vgg16
import vgg.basic_layer as bl
from constants import *
import math

def model(sess):
	images = tf.placeholder("float", [None, image_size, image_size, 3])
	bn = tf.placeholder(tf.bool)

	# vgg = vgg16.Vgg16()
	vgg = vgg16.VGGLayers(images, bn)

	h = [512, 1024, 1024,
		 256, 512,
		 128, 256,
		 128, 256]

	with tf.variable_scope("tb_extension"):
		
		c_ = classes+1

		out1 = bl.conv2d(vgg.conv4_3, 512, layer_boxes[0] * (c_ + 4), kernel=[1,5], bn=True, trainPhase=bn, name='out1')
		out2 = bl.conv2d(vgg.conv7, h[2], layer_boxes[1] * (c_ + 4), kernel=[1,5], bn=True, trainPhase=bn, name='out2')
		out3 = bl.conv2d(vgg.conv8_2, h[4], layer_boxes[2] * (c_ + 4), kernel=[1,5], bn=True, trainPhase=bn, name='out3')
		out4 = bl.conv2d(vgg.conv9_2, h[6], layer_boxes[3] * (c_ + 4), kernel=[1,5], bn=True, trainPhase=bn, name='out4')
		out5 = bl.conv2d(vgg.conv10_2, h[8], layer_boxes[4] * (c_ + 4), kernel=[1,5], bn=True, trainPhase=bn, name='out5')
		out6 = bl.conv2d(vgg.pool6, h[8], layer_boxes[5] * (c_ + 4), kernel=[1,1], bn=True, trainPhase=bn, name='out6')

	init = tf.global_variables_initializer()
	sess.run(init)

	outputs = [out1, out2, out3, out4, out5, out6]

	outfs = []
	for i, out in zip(range(len(outputs)), outputs):
		w = out.get_shape().as_list()[2]
		h = out.get_shape().as_list()[1]
		outf = tf.reshape(out, [-1, w*h*layer_boxes[i], c_ + 4])
		outfs.append(outf)
	
	formatted_outs = tf.concat(outfs, 1) # 23280 boxes flatted for all images
	print formatted_outs.shape
	pred_labels = formatted_outs[:, :, :c_]
	pred_locs = formatted_outs[:, :, c_:]
	
	return images, bn, outputs, pred_labels, pred_locs

def smooth_l1(x):
	l2 = 0.5 * (x**2.0)
	l1 = tf.abs(x) - 0.5

	condition = tf.less(tf.abs(x), 1.0)
	re = tf.where(condition, l2, l1)

	return re

def loss(pred_labels, pred_locs, total_boxes):
	positives = tf.placeholder(tf.float32, [None, total_boxes])
	negatives = tf.placeholder(tf.float32, [None, total_boxes])
	true_labels = tf.placeholder(tf.int32, [None, total_boxes])
	true_locs = tf.placeholder(tf.float32, [None, total_boxes, 4])

	posandnegs = positives + negatives
	print pred_labels.shape
	print true_labels.shape
	class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_labels, labels=true_labels) * posandnegs
	class_loss = tf.reduce_sum(class_loss, reduction_indices=1) / (1e-5 + tf.reduce_sum(posandnegs, reduction_indices=1))
	loc_loss = tf.reduce_sum(smooth_l1(pred_locs - true_locs), reduction_indices=2) * positives
	loc_loss = tf.reduce_sum(loc_loss, reduction_indices=1) / (1e-5 + tf.reduce_sum(positives, reduction_indices=1))
	total_loss = tf.reduce_mean(class_loss + tf.multiply(loc_loss, 1.0))

	return positives, negatives, true_labels, true_locs, total_loss, tf.reduce_mean(class_loss), tf.reduce_mean(loc_loss)

def box_scale(k):
	s_min = box_s_min
	s_max = 0.95
	m = 6.0

	s_k = s_min + (s_max - s_min) * (k - 1.0) / (m - 1.0) # equation 2

	return s_k

#caution: boxes should be stored in a numpy array, since using list will trigger segementation fault
'''
def default_boxes(out_shapes):
	boxes = np.ndarray(shape=(6, 38, 38, 12, 4), dtype=np.float64)
	for o_i in range(len(out_shapes)):
		layer_shape = out_shapes[o_i]
		s_k = box_scale(o_i + 1)	
		for y in range(layer_shape[2]):
			for x in range(layer_shape[1]):
				rs = box_ratios
				for i in range(len(rs)):
					scale = s_k
					default_w = scale * math.sqrt(rs[i])
					default_h = scale / math.sqrt(rs[i])
					c_x = (x + 0.5) / layer_shape[1]
					c_y = (y + 0.5) / layer_shape[2]
					boxes[o_i][x][y][2*i] = [c_y, c_x, default_w, default_h]
					c_x = (x + 1.0) / layer_shape[1]
					boxes[o_i][x][y][2*i + 1] = [c_y, c_x, default_w, default_h]

	return boxes
'''
def default_boxes(out_shapes):
	boxes = []

	for o_i in range(len(out_shapes)):
		layer_boxes = []
		layer_shape = out_shapes[o_i]
		s_k = box_scale(o_i + 1)
		for x in range(layer_shape[2]):
			x_boxes = []
			for y in range(layer_shape[1]):
				y_boxes = []
				rs = box_ratios
				for i in range(len(rs)):
					scale = s_k
					default_w = scale * np.sqrt(rs[i])
					default_h = scale / np.sqrt(rs[i])
					c_x = (x + 0.5) / float(layer_shape[2])
					c_y = (y + 0.5) / float(layer_shape[1])
					y_boxes.append([c_x, c_y, default_w, default_h])
					c_y = (y + 1.0) / float(layer_shape[1])
					y_boxes.append([c_x, c_y, default_w, default_h])
				x_boxes.append(y_boxes)
			layer_boxes.append(x_boxes)
		boxes.append(layer_boxes)
	return boxes