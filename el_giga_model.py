from keras import Input, Sequential
from keras.src.layers import Conv1D, Activation, MaxPooling1D, Flatten
from tensorflow.keras.layers import SimpleRNN, Dense, BatchNormalization, Dropout

import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')


class ElGigaModel:
    def __init__(self, config):
        self.INPUT_SIZE = config.input_size
        self.FILTER_LENGTH = config.filter_length
        self.KERNEL_SIZE = config.kernel_size
        self.DROP_RATE = config.drop_rate
        self.CLASSES = config.classes

    def build(self):
        model = Sequential([
            Input(shape=(self.INPUT_SIZE, 1), name='input'),
            Conv1D(filters=self.FILTER_LENGTH, kernel_size=self.KERNEL_SIZE, padding='same', strides=1, kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling1D(pool_size=1, strides=1),
            Dropout(self.DROP_RATE),
            SimpleRNN(128, input_shape=(len(self.CLASSES), 1), return_sequences=True),
            SimpleRNN(64, input_shape=(len(self.CLASSES), 1), return_sequences=True),
            SimpleRNN(64, input_shape=(len(self.CLASSES), 1), return_sequences=True),
            SimpleRNN(64, input_shape=(len(self.CLASSES), 1), return_sequences=True),
            SimpleRNN(32),
            Dense(len(self.CLASSES), activation='softmax'),
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model
