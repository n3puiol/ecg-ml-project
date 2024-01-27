import numpy as np
from matplotlib import pyplot as plt

from config import Config
from create_dataset import MitDBDataset


def plot_ecg(x, y, classes):
    plt.plot(x)
    plt.title("Class: " + classes[np.argmax(y)])
    plt.show()


if __name__ == '__main__':
    mit_dataset = MitDBDataset('./data/mitdb/1.0.0/', load=True)
    X, y = mit_dataset.load_dataset(feature='MLII')
    config = Config()
    plot_ecg(X[0], y[0], config.classes)
