# rom keras.layers import *
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.models import Model
from keras.layers import Input, PReLU, Dense, LSTM, multiply, concatenate, Activation
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from keras.callbacks import ModelCheckpoint
from tensorflow import keras
from tensorflow.keras import layers

def Model_setup(data):



    sys.path.insert(0, 'MLSTM-FCN/utils')
    #from layer_utils import AttentionLSTM




    #def generate_model():
        #ip = Input(shape=(valid_data.shape[1], valid_data.shape[2]))
    ip = keras.Input(shape=(data.shape[1], data.shape[2]))
    x = layers.Permute((2, 1))(ip)
    x = layers.LSTM(10)(x)
    x= layers.Dense(10, activation='relu')(x)
    x = layers.Dropout(0.8)(x)


    y= layers.Reshape((40,1,15))(ip)
    y= layers.DepthwiseConv2D(kernel_size=(9,1),depth_multiplier=20, data_format='channels_last', padding='valid' )(y)

    y = layers.Activation('relu')(y)
    # = layers.Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(ip)
    #y = layers.BatchNormalization()(y)
    #y= layers.Dropout(0.1)(y)
    y=layers.MaxPool2D(pool_size=(7,1), strides=(1,2),padding='valid')(y)
    #y = layers.Activation('elu')(y)
    #y= layers.Dropout(0.1)(y)
    #y = layers.BatchNormalization()(y)


    y= layers.DepthwiseConv2D(kernel_size=(5,1),depth_multiplier=10, data_format='channels_last', padding='valid' )(y)
    y = layers.Activation('relu')(y)


    #y=layers.MaxPool2D(pool_size=(3,1), strides=(1,2),padding='valid')(y)
    #y= layers.Dropout(0.1)(y)

    #y= layers.DepthwiseConv2D(kernel_size=(5,1),depth_multiplier=5, data_format='channels_last', padding='valid' )(y)
    #y = layers.Activation('relu')(y)
    # = layers.Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    #y = layers.BatchNormalization()(y)

    #y= layers.Dropout(0.2)(y)

    #y= layers.DepthwiseConv2D(kernel_size=(5,1),depth_multiplier=5, data_format='channels_last', padding='valid' )(y)
    #y = layers.Activation('relu')(y)
    #y = layers.BatchNormalization()(y)
    # = layers.Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)


    #y= layers.Dropout(0.1)(y)


    #y= layers.Reshape((60,15000))(y)
    print(y.shape)
    y= layers.Reshape((22,3000))(y)
    y = layers.GlobalAveragePooling1D()(y)
    y = layers.concatenate([x, y])
    y= layers.Dropout(0.1)(y)
    y1 = layers.Dense(1, activation='relu')(y)
    y2 = layers.Dense(1, activation='relu')(y)
    y3= layers.Dense(1, activation='relu')(y)
    y4= layers.Dense(1, activation='relu')(y)
    y5= layers.Dense(1, activation='relu')(y)
    y6= layers.Dense(1, activation='relu')(y)
    y7= layers.Dense(1, activation='relu')(y)
    y8= layers.Dense(1, activation='relu')(y)
    y9= layers.Dense(1, activation='relu')(y)
    y10= layers.Dense(1, activation='relu')(y)
    y11= layers.Dense(1, activation='relu')(y)
    y12= layers.Dense(1, activation='relu')(y)
    y13= layers.Dense(1, activation='relu')(y)
    y14= layers.Dense(1, activation='relu')(y)
    y15= layers.Dense(1, activation='relu')(y)
    y16= layers.Dense(1, activation='relu')(y)
    y17= layers.Dense(1, activation='relu')(y)
    y18= layers.Dense(1, activation='relu')(y)
    y19= layers.Dense(1, activation='relu')(y)
    y20= layers.Dense(1, activation='relu')(y)
    y21= layers.Dense(1, activation='relu')(y)
    y22= layers.Dense(1, activation='relu')(y)
    y23= layers.Dense(1, activation='relu')(y)
    y24= layers.Dense(1, activation='relu')(y)
    y25= layers.Dense(1, activation='relu')(y)
    y26= layers.Dense(1, activation='relu')(y)
    y27=layers.Dense(1, activation='relu')(y)
    y28= layers.Dense(1, activation='relu')(y)
    y29=layers.Dense(1, activation='relu')(y)
    y30= layers.Dense(1, activation='relu')(y)
    y31= layers.Dense(1, activation='relu')(y)
    y32= layers.Dense(1, activation='relu')(y)
    y=layers.concatenate([y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15,y16,y17,y18,y19,y20,y21,y22,y23,y24,y25,y26,y27,y28,y29,y30,y31,y32])
    #y= layers.Dropout(0.1)(y)
    out = layers.Dense(10, activation='softmax')(y)
    model=keras.Model(ip,out)
    return model
