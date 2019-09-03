from kt_model import HappyModel
from kt_utils import  data_preprocess, load_dataset
if __name__ == "__main__":
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()  # classes show how many gestures
    X_train, Y_train, X_test, Y_test = data_preprocess(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig)

    happyModel = HappyModel(X_train.shape[1:])
    happyModel.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    happyModel.fit(X_train, Y_train, epochs=40, batch_size=16)

    preds = happyModel.evaluate(X_test, Y_test, batch_size=50, verbose=1, sample_weight=None)

    print()
    print("Loss =  " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))