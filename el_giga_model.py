from keras import Input, Model
from keras.src.layers import TimeDistributed, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding


class ElGigaModel:
    def __init__(self, config):
        self.INPUT_SIZE = config.input_size
        self.FILTER_LENGTH = config.filter_length
        self.KERNEL_SIZE = config.kernel_size
        self.DROP_RATE = config.drop_rate

        self.classes = ['N', 'V', '/', 'A', 'F', '~']

    def build(self):
        inputs = Input(shape=(self.INPUT_SIZE, 1), name='input')
        rnn = SimpleRNN(128, input_shape=(self.INPUT_SIZE, 1), return_sequences=True)
        layer = rnn(inputs)
        layer = Conv1D(filters=self.FILTER_LENGTH,
                       kernel_size=self.KERNEL_SIZE,
                       padding='same',
                       strides=1,
                       kernel_initializer='he_normal')(layer)
        dense = Dense(len(self.classes), activation='softmax')

        outputs = TimeDistributed(dense)(layer)
        model = Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        print(model.summary())

        return model
