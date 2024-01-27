from __future__ import division, print_function
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, add, Flatten, Dropout, MaxPooling1D, Activation, BatchNormalization, \
    Lambda
from keras import backend as K
from keras.optimizers.legacy import Adam
from keras.src.layers import TimeDistributed


class BaselineModel:
    """
    implementation of the model in https://www.nature.com/articles/s41591-018-0268-3
    also have reference to codes at
    https://github.com/awni/ecg/blob/master/ecg/network.py
    and
    https://github.com/fernandoandreotti/cinc-challenge2017/blob/master/deeplearn-approach/train_model.py
    """

    def __init__(self, config):
        self.INPUT_SIZE = config.input_size
        self.FILTER_LENGTH = config.filter_length
        self.KERNEL_SIZE = config.kernel_size
        self.DROP_RATE = config.drop_rate
        self.CLASSES = config.classes

        # self.CLASSES = ['N', 'V', '/', 'A', 'F',
        #                 '~']  # ,'L','R',f','j','E','a']#,'J','Q','e','S'] are too few or not in the trainset, so excluded out

    def build(self):
        inputs = Input(shape=(self.INPUT_SIZE, 1), name='input')
        layer = self.first_conv_block(inputs)
        layer = self.main_loop_blocks(layer)
        return self.output_block(layer, inputs)

    def first_conv_block(self, inputs):
        layer = Conv1D(filters=self.FILTER_LENGTH,
                       kernel_size=self.KERNEL_SIZE,
                       padding='same',
                       strides=1,
                       kernel_initializer='he_normal')(inputs)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)

        shortcut = MaxPooling1D(pool_size=1,
                                strides=1)(layer)

        layer = Conv1D(filters=self.FILTER_LENGTH,
                       kernel_size=self.KERNEL_SIZE,
                       padding='same',
                       strides=1,
                       kernel_initializer='he_normal')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(self.DROP_RATE)(layer)
        layer = Conv1D(filters=self.FILTER_LENGTH,
                       kernel_size=self.KERNEL_SIZE,
                       padding='same',
                       strides=1,
                       kernel_initializer='he_normal')(layer)
        return add([shortcut, layer])

    def main_loop_blocks(self, layer):
        filter_length = self.FILTER_LENGTH
        n_blocks = 15
        for block_index in range(n_blocks):
            def zeropad(x):
                """
                zeropad and zeropad_output_shapes are from
                https://github.com/awni/ecg/blob/master/ecg/network.py
                """
                y = K.zeros_like(x)
                return K.concatenate([x, y], axis=2)

            def zeropad_output_shape(input_shape):
                shape = list(input_shape)
                assert len(shape) == 3
                shape[2] *= 2
                return tuple(shape)

            subsample_length = 2 if block_index % 2 == 0 else 1
            shortcut = MaxPooling1D(pool_size=subsample_length)(layer)

            # 5 is chosen instead of 4 from the original model
            if block_index % 4 == 0 and block_index > 0:
                # double size of the network and match the shapes of both branches
                shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)
                filter_length *= 2

            layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
            layer = Conv1D(filters=filter_length,
                           kernel_size=self.KERNEL_SIZE,
                           padding='same',
                           strides=subsample_length,
                           kernel_initializer='he_normal')(layer)
            layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
            layer = Dropout(self.DROP_RATE)(layer)
            layer = Conv1D(filters=filter_length,
                           kernel_size=self.KERNEL_SIZE,
                           padding='same',
                           strides=1,
                           kernel_initializer='he_normal')(layer)
            layer = add([shortcut, layer])
        return layer

    def output_block(self, layer, inputs):
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        # layer = Flatten()(layer)
        dense = Dense(len(self.CLASSES), activation='softmax')
        outputs = TimeDistributed(dense)(layer)
        model = Model(inputs=inputs, outputs=outputs)
        adam = Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
        model.compile(optimizer=adam,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        # model.summary()
        return model
