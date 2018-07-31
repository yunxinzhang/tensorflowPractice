import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

x = tf.constant(3)
y = tf.constant(4)

z = tf.Variable(0, name="lenght")
cnt = tf.Variable(0, name="counter")
print(z.name)


addOne = tf.assign(cnt, tf.add(cnt,1))
L2 = tf.assign(z, tf.add(tf.square(x), tf.square(y)))
# data type  int32,   cannt use tf.sqrt

with tf.Session() as sess:
	#tf.initialize_all_variables()   # depleted
	init = tf.global_variables_initializer()
	sess.run(init)                   # must run initializer
	for _ in range(5):
		sess.run(addOne)
		print(sess.run(cnt))
	sess.run(L2)
	print(sess.run(z))
