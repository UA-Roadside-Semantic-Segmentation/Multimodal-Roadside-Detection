import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, ReLU, Add, \
                                    Dropout, UpSampling2D, Cropping2D, Concatenate
from tensorflow.keras import Model, Input


def ASPPBlock(inputs, init_stride=1):
    xi = Conv2D(32, 3, padding='same', dilation_rate=1, strides=(1, init_stride))(inputs)
    xi = Activation('relu')(xi)
    xi = BatchNormalization()(xi)

    x1 = Conv2D(128, 3, padding='same', dilation_rate=(1,  6))(xi)
    x1 = Activation('relu')(x1)
    x1 = BatchNormalization()(x1)

    x2 = Conv2D(128, 3, padding='same', dilation_rate=(1, 12))(xi)
    x2 = Activation('relu')(x2)
    x2 = BatchNormalization()(x2)

    x3 = Conv2D(128, 3, padding='same', dilation_rate=(1, 24))(xi)
    x3 = Activation('relu')(x3)
    x3 = BatchNormalization()(x3)

    x4 = Conv2D(128, 3, padding='same', dilation_rate=(2, 1))(xi)
    x4 = Activation('relu')(x4)
    x4 = BatchNormalization()(x4)

    x5 = Conv2D(128, 3, padding='same', dilation_rate=(4, 1))(xi)
    x5 = Activation('relu')(x5)
    x5 = BatchNormalization()(x5)

    concat = Concatenate()([x1, x2, x3, x4, x5])

    x = Conv2D(32, 3, padding='same', dilation_rate=1)(concat)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Add()([x, xi])

    return x


def SkipBlock(inputs, init_stride=1):
    xi = Conv2D(32, 3, padding='same', dilation_rate=1, strides=(1, init_stride))(inputs)
    xi = Activation('relu')(xi)
    xi = BatchNormalization()(xi)

    x = Conv2D(32, 3, padding='same', dilation_rate=1)(xi)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(32, 3, padding='same', dilation_rate=1)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(32, 3, padding='same', dilation_rate=1)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(32, 3, padding='same', dilation_rate=1)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Add()([x, xi])

    return x


def create_model(scan_lines):
    inputs = Input((scan_lines, 2048, 4))
    x = inputs #ReLU(max_value=1000)(inputs)

    x = Conv2D(32, 3, padding='same', dilation_rate=1)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = SkipBlock(x, init_stride=1)
    x = SkipBlock(x, init_stride=2)
    x = SkipBlock(x, init_stride=2)
    x = SkipBlock(x, init_stride=2)

    x = ASPPBlock(x, init_stride=1)

    x = UpSampling2D(size=(1, 2), interpolation='bilinear')(x)

    x = SkipBlock(x, init_stride=1)

    x = UpSampling2D(size=(1, 2), interpolation='bilinear')(x)

    x = SkipBlock(x, init_stride=1)

    x = UpSampling2D(size=(1, 2), interpolation='bilinear')(x)

    x = Dropout(0.1)(x)

    x = Conv2D(32, 3, padding='same', dilation_rate=1)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(16, 3, padding='same', dilation_rate=1)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(2, 1, padding='same', dilation_rate=1)(x)

    output = Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=output)

    return model

