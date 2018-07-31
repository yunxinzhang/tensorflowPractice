import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# y = 2*x + 1

x = np.linspace(0,10,100)
y = x*2 + 1 + np.random.uniform(-0.5,0.5,100)

plt.scatter(x,y)

w = tf.Variable(tf.random_uniform([1],-5,5))
b = tf.Variable(tf.zeros([1]))
yp = w*x +b

loss = tf.reduce_mean(tf.square(y-yp))

with tf.Session() as sess:
	optimizer = tf.train.GradientDescentOptimizer(0.002)
	train = optimizer.minimize(loss)
	init = tf.global_variables_initializer()
	sess.run(init)
	for i in range(1000):
		sess.run(train)
		print(sess.run(loss))
	print(sess.run(w), sess.run(b))
	plt.plot(x, x*sess.run(w)+sess.run(b))
	plt.show()
