from kt_model import HappyModel
from kt_utils import  data_preprocess, load_dataset
from keras.preprocessing import image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

if __name__ == "__main__":
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()  # classes show how many gestures
    X_train, Y_train, X_test, Y_test = data_preprocess(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig)

    happyModel = HappyModel(X_train.shape[1:])
    happyModel.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    happyModel.fit(X_train, Y_train, epochs=50, batch_size=16)

    preds = happyModel.evaluate(X_test, Y_test, batch_size=50, verbose=1, sample_weight=None)

    print()
    print("Loss =  " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))

    img_path = "images/happy_test1.jpg"
    img = image.load_img(img_path, target_size=(64, 64))
    imshow(img)
    plt.show()

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print(happyModel.predict(x))

    happyModel.summary()

    # plot_model(happyModel, to_file='HappyModel.png')
    # SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))