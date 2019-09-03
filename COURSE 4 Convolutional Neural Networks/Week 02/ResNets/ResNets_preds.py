from keras.models import Model, load_model
from keras.preprocessing import image
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

if __name__ == "__main__":
    model = load_model('ResNet50.h5')

    img_path = 'images/my_image.jpg'
    my_image = scipy.misc.imread(img_path)
    imshow(my_image)
    plt.show()

    img = image.load_img(img_path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)
    print("class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ")
    print(model.predict(x))