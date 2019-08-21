from cnn_utils import load_dataset, data_preprocess, random_mini_batches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

"""
the model is INPUT -> CONV -> RELU -> POOL -> CONV -> RELU -> POOL -> FC -> SOFTMAX
"""

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, shape = [None, n_H0, n_W0, n_C0], name="X")
    Y = tf.placeholder(tf.float32, shape = [None, n_y], name="Y")
    return X, Y


def initialize_parameters():
    tf.set_random_seed(1)
    # W1 = tf.Variable("W1", [4, 4,3,8], initializer= tf.contrib.layers.xavier_initializer(seed = 0))
    # W2 = tf.Variable("W2", [2,2,8,16], initializer= tf.contrib.layers.xavier_initializer(seed = 0))
    #
    # b1 = tf.Variable("b1", [1,1,1,8], initializer= tf.contrib.layers.xavier_initializer(seed = 0))
    # b2 = tf.Variable("b2", [1, 1, 1,16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W1 = tf.Variable(tf.random_normal([4, 4, 3, 8]), "W1")
    W2 = tf.Variable(tf.random_normal([2, 2, 8, 16]), "W2")
    b1 = tf.Variable(tf.random_normal([1, 1, 1, 8]), "b1")
    b2 = tf.Variable(tf.random_normal([1, 1, 1, 16]), "b2")
    parameters = {
        "W1": W1,
        "W2": W2,
        "b1": b1,
        "b2": b2
                  }
    return parameters


def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    Z1 = tf.nn.conv2d(X, W1, [1, 1, 1, 1], padding='SAME')
    # convolute here
    A1 = tf.nn.relu(Z1 + b1)  # Z1.shape: [None,64,64,8]
    P1 = tf.nn.max_pool(A1 , ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding="SAME")

    Z2 = tf.nn.conv2d(P1, W2, [1, 1, 1, 1], padding='SAME')
    A2 = tf.nn.relu(Z2 + b2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

    P2 = tf.contrib.layers.flatten(P2)

    Z3 = tf.contrib.layers.fully_connected(P2, 6, activation_fn=None, scope='fc1')

    return Z3


def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009, num_epochs = 200, minibatch_size = 64, print_cost = True):

    tf.reset_default_graph()

    tf.set_random_seed(1)
    seed = 3
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_Y = Y_train.shape[1]
    costs = []

    X, Y = create_placeholders(64, 64, 3, 6)

    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)

    cost = compute_cost(Z3, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):

            minibatch_cost = 0.

            #  mini - batch
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch

                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                minibatch_cost += temp_cost / num_minibatches

            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i is: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1== 0:
                costs.append(minibatch_cost)

        plt.plot(np.squeeze(costs))
        plt.xlabel("iterations(per fives)")
        plt.ylabel("cost")
        plt.title("learning rate =" + str(learning_rate))
        plt.show()

        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})

        print("Train accuracy :" + str(train_accuracy))
        print("Test accuracy : " + str(test_accuracy))

        # save model
        saver.save(sess, './checkpoint_dir/MyModel', global_step=num_epochs, write_meta_graph=True)
        sess.close()
        return train_accuracy, test_accuracy, parameters


def get_trained_parameters():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()  # classes show how many gestures
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = data_preprocess(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig)

    # show the source image from the training set
    # index = 6
    # plt.imshow(X_train_orig[index])
    # plt.show()
    # print("Y = :" + str(np.squeeze(Y_train_orig[:, index])))

    _, _, parameters = model(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig)
    return parameters


if __name__ == "__main__":
    get_trained_parameters()
