from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint

from baseline_model import BaselineModel
from create_dataset import MitDBDataset
from el_giga_model import ElGigaModel


class Config:
    def __init__(self):
        self.input_size = 256
        self.feature = "MLII"
        self.filter_length = 32
        self.kernel_size = 16
        self.drop_rate = 0.2
        self.epochs = 10
        self.batch = 256
        self.patience = 10


def train():
    config = Config()
    # model = BaselineModel(config).build()
    model = ElGigaModel(config).build()

    mit_dataset = MitDBDataset('./data/mitdb/1.0.0/', load=True)
    X, y = mit_dataset.load_dataset(feature=config.feature)
    Xe = np.expand_dims(X, axis=2)
    Xe, X_val, y, y_val = train_test_split(Xe, y, test_size=0.2, random_state=1)

    callbacks = [
        EarlyStopping(patience=config.patience, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.01, verbose=1),
        TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_grads=False, write_images=True),
        ModelCheckpoint('models/{}-latest.hdf5'.format(config.feature), monitor='val_loss', save_best_only=False,
                        verbose=1, save_freq=10)
    ]
    y = np.expand_dims(y, axis=2)
    y_val = np.expand_dims(y_val, axis=2)
    y = np.swapaxes(y, 1, 2)
    y_val = np.swapaxes(y_val, 1, 2)

    model.fit(Xe, y,
              validation_data=(X_val, y_val),
              epochs=config.epochs,
              batch_size=config.batch,
              callbacks=callbacks,
              initial_epoch=0)


if __name__ == '__main__':
    train()
