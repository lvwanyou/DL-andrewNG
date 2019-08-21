from cnn_utils import load_image
import tensorflow as tf

## START CODE HERE ## (PUT YOUR IMAGE NAME)
# my_image = "thumbs_up.jpg"





if __name__ == "__main__":

    my_image = load_image()

    saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel-200.meta')
    graph = tf.get_default_graph()

    pred = graph.get_tensor_by_name("fc1/BiasAdd:0")
    X = graph.get_tensor_by_name("X:0")

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir'))

        ret = sess.run(pred, feed_dict={X: my_image})

        num_pred = sess.run(tf.argmax(ret, 1))
        print("predicted num is : %d" % num_pred)
        sess.close()