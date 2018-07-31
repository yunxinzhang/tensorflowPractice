# -*- coding: UTF-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

x = tf.constant([3,4,5], name='x')

y = tf.constant([1,2,3], name='y')

z = tf.add(x,y, name='z')

w = x * y
w = w - z
path = './log'
with tf.Session() as sess:
	with tf.summary.FileWriter(path, sess.graph) as writer:
		a , b  = sess.run([z ,w ])
		#c = sess.run(w-z)
#from google.datalab.ml import TensorBoard

#TensorBoard().start("./log")
