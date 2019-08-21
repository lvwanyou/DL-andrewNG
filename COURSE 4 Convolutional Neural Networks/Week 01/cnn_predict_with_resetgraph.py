from cnn_model import create_placeholders, forward_propagation, initialize_parameters
import tensorflow as tf
from cnn_utils import load_image

if __name__ == "__main__":

    my_image = load_image()

    tf.set_random_seed(1)
    X, Y = create_placeholders(my_image.shape[1], my_image.shape[2], my_image.shape[3], 6)

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

    pred = forward_propagation(X, parameters)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir'))

        ret = sess.run(pred, feed_dict={X: my_image})
        num_pred = sess.run(tf.argmax(ret, 1))

        print("predicted num is : %d" % num_pred)
        sess.close()