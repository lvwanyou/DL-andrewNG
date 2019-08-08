import tensorflow as tf
import numpy as np
coefficients = np.array([[1.0], [-10.0], [25.]])

w = tf.Variable(0, dtype=tf.float32)    # tf.constant( , )
x = tf.placeholder(tf.float32, [3, 1])

# cost = tf.add(tf.add(w ** 2, tf.multiply(-10., w)), 25)
cost = x[0][0] * w ** 2 + x[1][0] * w + x[2][0]
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)      # back propagation

init = tf.initialize_all_variables()
session = tf.Session()
session.run(init)
print(session.run(w))  # print weight without training it

# with GD method & iteration for 1000 times
for i in range(1000):
    session.run(train, feed_dict={x: coefficients})
    if i % 100 == 0:
        print("cost after iteration %d : %f " % (i, session.run(w)))

print(session.run(w))

