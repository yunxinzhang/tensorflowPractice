# -*- coding: UTF-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

def calc_area(triangle):
	s = (triangle[0] + triangle[1] + triangle[2])/2
	return tf.sqrt(s*(s-triangle[0])*(s-triangle[1])*(s-triangle[2]))


with tf.Session() as sess:
	tr1 = tf.constant([3,4,5.0])
	a = calc_area(tr1)
	print(sess.run(a))
