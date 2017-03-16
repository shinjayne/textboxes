import tensorflow as tf
import numpy as np

def weightVariable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1, name='weights')
	return tf.Variable(initial)

def biasVariable(shape):
	initial = tf.constant(0.1, shape=shape, name='biases')
	return tf.Variable(initial)

def maxPool(input, stride=2, kernel=2, padding='SAME', name='pool'):
	return tf.nn.max_pool(input, ksize=[1, kernel, kernel, 1], strides=[1, stride, stride, 1], padding='SAME', name=name)

def conv2d(input, inputNum, outputNum, kernel=[3, 3], strides=[1, 1], padding='SAME', bn=False, trainPhase=True, name='conv2d'):
	with tf.name_scope(name) as scope:
		W = weightVariable([kernel[0], kernel[1], inputNum, outputNum])
		b = biasVariable([outputNum])
		conv_out = tf.nn.conv2d(input, W, strides=[1, strides[0], strides[1], 1], padding=padding)
		biased_out = tf.nn.bias_add(conv_out, b)
		out = tf.nn.relu(biased_out)
		if bn:
			out = tf.contrib.layers.batch_norm(out, center=False, is_training=trainPhase)
		return out
