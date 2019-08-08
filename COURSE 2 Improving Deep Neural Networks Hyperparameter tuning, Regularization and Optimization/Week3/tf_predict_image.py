import scipy
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from tf_utils import predict
from tf_program_assignment import get_trained_parameters
## START CODE HERE ## (PUT YOUR IMAGE NAME)
# my_image = "thumbs_up.jpg"
my_image_test1 = "thumbs_three.jpg"
## END CODE HERE ##

# We preprocess your image to fit your algorithm.
fname = "images/" + my_image_test1
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T

parameters = get_trained_parameters()
my_image_prediction = predict(my_image, parameters)

plt.imshow(image)
plt.show()
print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))