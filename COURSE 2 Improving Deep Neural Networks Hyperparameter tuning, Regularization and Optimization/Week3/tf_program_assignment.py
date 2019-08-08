import numpy as np
import matplotlib.pyplot as plt
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
import tensorflow as tf
from tensorflow.python.framework import ops
# the model is LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX


def create_placeholder(n_x, n_y):
    """
    create placeholder for the tensorflow session
    :param n_x: features_dimension of x
    :param n_y: features_dimension of y
    :return: X,Y
    """
    X = tf.placeholder(tf.float32, shape=[n_x, None])
    Y = tf.placeholder(tf.float32, shape=[n_y, None])
    return X, Y

def initializer_parameter():
    W1 = tf.get_variable("W1", [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer)
    W2 = tf.get_variable("W2",[12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer)
    W3 = tf.get_variable("W3", [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer)

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3
    }
    return parameters


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    return Z3


def computer_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, num_epochs=1500, minibatch_size=32, print_cost= True):
    ops.reset_default_graph()
    tf.set_random_seed(seed=1)
    seed = 3
    parameters = initializer_parameter()

    n_x = X_train.shape[0]
    m = X_train.shape[1]
    n_y = Y_train.shape[0]
    X, Y = create_placeholder(n_x, n_y)

    costs = []

    Z3 = forward_propagation(X, parameters)
    cost = computer_cost(Z3, Y)

    # backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibtches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                minibatch_x, minibatch_y = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_x, Y: minibatch_y})

                epoch_cost += minibatch_cost / num_minibtches

            if print_cost is True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost is True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters


def get_trained_parameters():   # if __name__ == '__main__':
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    # Example of a picture   X_train_orig.shape(1080, 64, 64, 3); Y_train_orig.shape(1, 1080)
    # index = 0
    # plt.imshow(X_train_orig[index])
    # plt.show()
    # print("y = " + str(np.squeeze(Y_train_orig[:, index])))


    # Flatten the training and test image
    # one shape dimension can be -1. In this case, the value is inferred from the length of the array
    X_train_orig = X_train_orig.reshape(X_train_orig.shape[0], -1).T  # shape(12288, 1080)
    X_test_orig = X_test_orig.reshape(X_test_orig.shape[0], -1).T

    # Normalize image vectors
    X_train = X_train_orig / 255
    X_test = X_test_orig / 255

    # convert Y training and test to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, 6)  # detect gesture from 0 to five
    Y_test = convert_to_one_hot(Y_test_orig, 6)
    parameters = model(X_train, Y_train, X_test, Y_test)

    return parameters


