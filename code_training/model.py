import tensorflow as tf
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.engine.saving import load_model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, TimeDistributed, LSTM, Dropout, Dense, BatchNormalization, \
    LeakyReLU, Conv1D, MaxPooling1D,Input
from tensorflow.keras.regularizers import l2

def build_CRNN_model():
    input_shape = (3,1,20)
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=5, activation="relu",padding="same",strides=1 ,input_shape=(None, None, 20)),input_shape=input_shape))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(units=32, return_sequences=True, kernel_initializer='random_uniform'))
    model.add(TimeDistributed(Dropout(rate=0.4)))
    model.add(LSTM(units=32, return_sequences=True, kernel_initializer='random_uniform'))
    model.add(TimeDistributed(Dropout(rate=0.4)))
    model.add(TimeDistributed(Dense(units=256)))
    model.add(TimeDistributed(LeakyReLU(alpha=0.1)))
    model.add(TimeDistributed(Dropout(rate=0.5)))
    model.add(TimeDistributed(Dense(units=3)))

    return model

def build_RNN_model():
    model = Sequential()
    model.add(LSTM(units=32, input_shape=(3, 20), return_sequences=True, kernel_initializer='random_uniform'))
    model.add(Dropout(rate=0.4))
    model.add(LSTM(units=32, input_shape=(3, 20), return_sequences=True, kernel_initializer='random_uniform'))
    model.add(Dropout(rate=0.4))
    model.add(LSTM(units=32, return_sequences=True, kernel_initializer='random_uniform'))
    model.add(Dropout(rate=0.4))
    model.add(Dense(units=256))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
    model.add(Dense(units=3))
    # model.summary()
    return model