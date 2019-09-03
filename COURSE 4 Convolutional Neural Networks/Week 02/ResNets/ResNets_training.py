from ResNets_model import ResNet50
import keras.backend as K
from resnets_utils import data_preprocess, load_dataset

if __name__ == "__main__":
    K.set_image_data_format('channels_last')
    K.set_learning_phase(1)

    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()  # classes show how many gestures
    X_train, Y_train, X_test, Y_test = data_preprocess(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig)

    ResNets_model = ResNet50(input_shape=(64, 64, 3), classes=6)
    ResNets_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
    ResNets_model.fit(X_train, Y_train, epochs=2, batch_size=2)
    preds = ResNets_model.evaluate(X_test, Y_test)

    print()
    print("Loss =  " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))
    ResNets_model.save("ResNet50.h5")