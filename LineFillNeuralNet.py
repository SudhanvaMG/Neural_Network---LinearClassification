import tensorflow as tf
import numpy as np

#input data (100 dummy data) to fill the graph
x_data = np.float32(np.random.rand(2,100))
y_data = np.dot([0.1, 0.2], x_data) + 0.3

#constructing a linear model
b = tf.Variable(tf.zeros(1))
W = tf.Variable(tf.random_uniform([1,2], -1, 1))
y = tf.matmul(W, x_data) + b

#gradient descent time - optimizer function
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

#init the Variables
init = tf.initialize_all_variables()

#launch the graph
sess = tf.Session()
sess.run(init)

#train - to fit the plane
for step in xrange(0, 200):
    sess.run(train)
    if step % 20 == 0:
        print step , sess.run(W) , sess.run(b)
