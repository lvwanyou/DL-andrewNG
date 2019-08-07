import numpy as np
import math
np.random.seed(3)
def random_mini_batches_test_case():
    X_assess = np.random.rand(12288, 148) - 0.5
    Y_assess = np.random.rand(1, 148) - 0.5 > 0
    mini_batch_size = 64
    return X_assess, Y_assess, mini_batch_size

def random_mini_batches(X, Y, mini_batch_size):
    m = X.shape[1]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    """
    shuffle synchronously between X and Y
    """
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, mini_batch_size * k: mini_batch_size * (k + 1)]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * k: mini_batch_size * (k + 1)]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        # START CODE HERE  (approx. 2 lines)
        mini_batch_X = shuffled_X[:, mini_batch_size * num_complete_minibatches: m]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * num_complete_minibatches: m]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def random_mini_batches_print():
    X_assess, Y_assess, mini_batch_size = random_mini_batches_test_case()
    mini_batches = random_mini_batches(X_assess, Y_assess, mini_batch_size)

    print("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
    print("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
    print("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
    print("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
    print("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape))
    print("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
    print("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))


