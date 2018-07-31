import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf


x1 = tf.constant([3,4,5])
x2 = tf.constant([1,2,3])
x3 = tf.constant([2,4,6])   # all np.array

x4 = tf.add(x1,x2)
x5 = x2 * x3
a, b = None, None
with tf.Session() as sess:
	a, b = sess.run([x4, x5])  # args can be a list

print(a,b)
print(type(a))
print(type(b))
