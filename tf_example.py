import tensorflow as tf
import numpy as np

w = tf.Variable(0, dtype=tf.float32)
cost = tf.add(tf.add(w ** 2, tf.multiply(-10., w)), 25)
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.initialize_all_variables()
session = tf.Session()
session.run(init)
print(session.run(w))  # print weight without training it

# with GD method & iteration for 1000 times
for i 
