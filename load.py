import numpy as np

from config import Config
from create_dataset import MitDBDataset
import tensorflow as tf

from plot_data import plot_ecg

tf.config.set_visible_devices([], 'GPU')


def load():
    mit_dataset = MitDBDataset('./data/mitdb/1.0.0/', load=True)
    config = Config()
    X, y = mit_dataset.load_dataset(feature='MLII')

    model = tf.keras.models.load_model('models/MLII-latest.hdf5')
    y_pred = model.predict(X)
    num_correct = 0
    num_incorrect = 0
    for i in range(len(y_pred)):
        max_idx = np.argmax(y_pred[i])
        if np.isclose(y[i][max_idx], 1.0):
            num_correct += 1
        else:
            num_incorrect += 1

    print("Correct: ", num_correct)
    print("Incorrect: ", num_incorrect)
    print("Accuracy: ", num_correct / (num_correct + num_incorrect))



    # max_idx = np.argmax(y_pred[0])



    # if np.isclose(y[0][max_idx], 1.0):
    #     print("Correct")
    #     # plot_ecg(X[1], y[1], config.classes)
    # else:
    #     print("Incorrect")



if __name__ == '__main__':
    load()
