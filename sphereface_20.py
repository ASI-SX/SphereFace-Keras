import keras
from keras.layers import Convolution2D, Add, Activation, PReLU, Dense, Input
from keras.initializers import TruncatedNormal
from keras.engine.topology import Layer
import MarginInnerProductLayer as MIPL
from keras import backend as K
from keras.models import Model

def modeldef(img_rows, img_cols, color_type=1, num_classes=None):
    
    global bn_axis

    if K.image_dim_ordering() == 'tf':
        
      bn_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type))

    else:
      bn_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols))

    #Conv 1.X

    conv1_1 = Convolution2D(64, 3, strides=2, padding='valid', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')
    relu1_1 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv1_1)

    conv1_2 = Convolution2D(64, 3, strides=1, padding='valid', kernel_initializer=TruncatedNormal(stddev=0.01), bias_initializer='zeros')(relu1_1)
    relu1_2 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv1_2)

    conv1_3 = Convolution2D(64, 3, strides=1, padding='valid', kernel_initializer=TruncatedNormal(stddev=0.01), bias_initializer='zeros')(relu1_2)
    relu1_3 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv1_3)

    res1_3 = Add()([relu1_1,relu1_3])

    #Conv 2.X

    conv2_1 = Convolution2D(128, 3, strides=2, padding='valid', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(res1_3)
    relu2_1 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv2_1)

    conv2_2 = Convolution2D(128, 3, strides=1, padding='valid', kernel_initializer=TruncatedNormal(stddev=0.01), bias_initializer='zeros')(relu2_1)
    relu2_2 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv2_2)

    conv2_3 = Convolution2D(128, 3, strides=1, padding='valid', kernel_initializer=TruncatedNormal(stddev=0.01), bias_initializer='zeros')(relu2_2)
    relu2_3 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv2_3)

    res2_3 = Add()([relu2_1,relu2_3])

    conv2_4 = Convolution2D(128, 3, strides=1, padding='valid', kernel_initializer=TruncatedNormal(stddev=0.01), bias_initializer='zeros')(res2_3)
    relu2_4 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv2_4)

    conv2_5 = Convolution2D(128, 3, strides=1, padding='valid', kernel_initializer=TruncatedNormal(stddev=0.01), bias_initializer='zeros')(relu2_4)
    relu2_5 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv2_5)

    res2_5 = Add()([res2_3,relu2_5])

    #Conv 3.X

    conv3_1 = Convolution2D(256, 3, strides=2, padding='valid', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(res2_5)
    relu3_1 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv3_1)

    conv3_2 = Convolution2D(256, 3, strides=1, padding='valid', kernel_initializer=TruncatedNormal(stddev=0.01), bias_initializer='zeros')(relu3_1)
    relu3_2 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv3_2)

    conv3_3 = Convolution2D(256, 3, strides=1, padding='valid', kernel_initializer=TruncatedNormal(stddev=0.01), bias_initializer='zeros')(relu3_2)
    relu3_3 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv3_3)

    res3_3 = Add()([relu3_1,relu3_3])

    conv3_4 = Convolution2D(256, 3, strides=1, padding='valid', kernel_initializer=TruncatedNormal(stddev=0.01), bias_initializer='zeros')(res3_3)
    relu3_4 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv3_4)

    conv3_5 = Convolution2D(256, 3, strides=1, padding='valid', kernel_initializer=TruncatedNormal(stddev=0.01), bias_initializer='zeros')(relu3_4)
    relu3_5 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv3_5)

    res3_5 = Add()([res3_3,relu3_5])

    conv3_6 = Convolution2D(256, 3, strides=1, padding='valid', kernel_initializer=TruncatedNormal(stddev=0.01), bias_initializer='zeros')(res3_5)
    relu3_6 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv3_6)

    conv3_7 = Convolution2D(256, 3, strides=1, padding='valid', kernel_initializer=TruncatedNormal(stddev=0.01), bias_initializer='zeros')(relu3_6)
    relu3_7 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv3_7)

    res3_7 = Add()([res3_5,relu3_7])

    conv3_8 = Convolution2D(256, 3, strides=1, padding='valid', kernel_initializer=TruncatedNormal(stddev=0.01), bias_initializer='zeros')(res3_7)
    relu3_8 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv3_8)

    conv3_9 = Convolution2D(256, 3, strides=1, padding='valid', kernel_initializer=TruncatedNormal(stddev=0.01), bias_initializer='zeros')(relu3_8)
    relu3_9 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv3_9)

    res3_9 = Add()([res3_7,relu3_9])

    #Conv 4.X

    conv4_1 = Convolution2D(512, 3, strides=2, padding='valid', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(res3_9)
    relu4_1 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv4_1)

    conv4_2 = Convolution2D(512, 3, strides=1, padding='valid', kernel_initializer=TruncatedNormal(stddev=0.01), bias_initializer='zeros')(relu4_1)
    relu4_2 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv4_2)

    conv4_3 = Convolution2D(512, 3, strides=1, padding='valid', kernel_initializer=TruncatedNormal(stddev=0.01), bias_initializer='zeros')(relu4_2)
    relu4_3 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(conv4_3)

    res4_3 = Add()([relu4_1,relu4_3])

    #FC5

    fc5 = Dense(512, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(res4_3)

    #FC6

    fc6 = MIPL.MarginInnerProductLayer(512,num_classes)

    model = model = Model(img_input, fc6)

    return model