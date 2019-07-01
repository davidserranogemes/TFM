from keras.datasets import mnist
from autokeras import ImageClassifier
import numpy as np

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    """
    x_train = np.load("x_train.npy")
    x_test = np.load("x_test.npy")
    y_train = np.load("y_train.npy")
    y_test = np.load("y_test.npy")
    """
    print("Datos cargados\n")
    x_train = x_train.reshape(x_train.shape+(1,))
    x_test = x_test.reshape(x_test.shape+(1,))
    print("Reshape realizados\n")
    clf = ImageClassifier(verbose=True, augment=False)
    print("ImageClassifier creado\n")
    clf.fit(x_train, y_train, time_limit=1*60*60)
    print("Fit inicial creado\n")
    clf.final_fit(x_train, y_train, x_test, y_test, retrain=False)
    print("Reentrenando el modelo elegido\n")
    y = clf.evaluate(x_test, y_test)
    print("evaluado\n")
    print(y * 100)
