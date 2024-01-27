from keras import Input, Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, BatchNormalization

import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')


class ElGigaModel:
    def __init__(self, config):
        self.INPUT_SIZE = config.input_size
        self.FILTER_LENGTH = config.filter_length
        self.KERNEL_SIZE = config.kernel_size
        self.DROP_RATE = config.drop_rate

        self.classes = ['N', 'V', '/', 'A', 'F', '~']

    def build(self):
        model = Sequential([
            Input(shape=(self.INPUT_SIZE, 1), name='input'),
            SimpleRNN(128, input_shape=(len(self.classes), 1), return_sequences=True),
            SimpleRNN(128),
            Dense(len(self.classes), activation='softmax'),
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model
