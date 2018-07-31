import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)

res = tf.multiply(x1,x2)

# tf.mul     Don't have this function. 

with tf.Session() as sess:
	t1 = np.random.random()
	t2 = np.random.random()
	print(t1,t2)
# feed_dict
	print(sess.run(res, feed_dict={x1:t1, x2:t2}))
