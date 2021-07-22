import datetime
import pickle
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.engine.saving import load_model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, TimeDistributed, LSTM, Dropout, Dense, BatchNormalization, \
    LeakyReLU, Conv1D, MaxPooling1D,Input
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def build_CRNN_model():
    input_shape = (3,1,20)
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=5, activation="relu",padding="same",strides=1 ,input_shape=(None, None, 20)),input_shape=input_shape))
    # model.add ((Conv1D(filters=64, kernel_size=4, activation="relu",input_shape=(None,45)) ))
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
    model.add(LSTM(units=32, input_shape=(timesteps, 20), return_sequences=True, kernel_initializer='random_uniform'))
    model.add(Dropout(rate=0.4))
    model.add(LSTM(units=32, input_shape=(timesteps, 20), return_sequences=True, kernel_initializer='random_uniform'))
    model.add(Dropout(rate=0.4))
    model.add(LSTM(units=32, return_sequences=True, kernel_initializer='random_uniform'))
    model.add(Dropout(rate=0.4))
    model.add(Dense(units=256))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
    model.add(Dense(units=3))
    # model.summary()
    return model

def load_slding_data(timesteps,name):
    X = []
    Y = []
    Gesture = ["circle", "eight", "rectangle", "up", "down", "left", "right"]
    # Gesture = ["up", "down", "left", "right"]
    path = "D:/thumouse_training_data/" + name + "/sliding_data/"
    print("load : " + path)
    for i in Gesture:
        for j in range(2, 4):

            cam_voxel = np.load(path+ i +"_time" + str(j) + "_cam.npy", allow_pickle=True)
            radar_voxel = np.load(path+ i +"_time" + str(j) + "_radar_dict.npy", allow_pickle=True)
            if timesteps == 3:
                X.append(radar_voxel[118:])
                # Y.append(cam_voxel[115:-3,:])
                Y.append(cam_voxel[118:,:])
            elif timesteps == 12:
                X.append(radar_voxel[109:])
                Y.append(cam_voxel[109:,:])
            # print(np.shape(cam_voxel))
            # print(np.shape(radar_voxel))

    X = np.reshape(X, [-1, timesteps, 20])
    Y = np.reshape(Y, [-1, timesteps, 3])
    print(np.shape(X))
    print(np.shape(Y))
    X = np.asarray(X)
    Y = np.asarray(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=3, shuffle=True)
    return   X_train, X_test, Y_train, Y_test

is_use_pre_train = False
epochs = 50000 # for ealry stop setting
# epochs = 300 # for quickly run out result
timesteps = 3
date = 121319
# X_train, X_test, Y_train, Y_test = load_none_slding_data(timesteps)
model   = ["CRNN","RNN"]
Gesture = ["circle", "eight", "rectangle", "up", "down", "left", "right"]
dataset = ["transfer_0", "transfer_0dot05", "transfer_0dot2", "transfer_0dot5", "transfer_0dot8", "transfer_0dot95",
           "transfer_1"]


for model_type in model:
    for data_set in dataset:
        X_train, X_test, Y_train, Y_test = load_slding_data(timesteps,data_set)

        if model_type == 'CRNN':
            X_train = np.expand_dims(X_train, axis=2)
            X_test = np.expand_dims(X_test, axis=2)
        print(np.shape(X_train))
        print(np.shape(X_test))
        print(np.shape(Y_train))
        print(np.shape(Y_test))

        if not is_use_pre_train:
            print('Building Model...')
            if model_type == "CRNN":
                model = build_CRNN_model()
            elif model_type == "RNN":
                model = build_RNN_model()

            sgd = optimizers.SGD(lr=5e-5, momentum=0.9, decay=1e-6, nesterov=True)
            adam = Adam(lr=1e-6, decay=1e-6)
            model.compile(optimizer=adam, loss='mean_squared_error')

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)# 當val_loss最小50Epoch 都不變
        # mc = ModelCheckpoint(
        #     'D:/pythonProject/ML_thumouse/' + str(datetime.datetime.now()).replace(':', '-').replace(
        #         ' ', '_') + '.h5',
        #     monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        if model_type == "CRNN":
            mc = ModelCheckpoint(
                'D:/pythonProject/ML_thumouse/' + data_set+ "_CRNN" + '.h5',
                monitor='val_loss', mode='min', verbose=1, save_best_only=True)

        elif model_type == "RNN":
            mc = ModelCheckpoint(
                'D:/pythonProject/ML_thumouse/' + data_set + "_RNN" +'.h5',
                monitor='val_loss', mode='min', verbose=1, save_best_only=True)

        history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                            # batch_size=12, epochs=epochs,callbacks=[mc])
                            batch_size=12, epochs=epochs,  callbacks=[es, mc]) # early stop + check point


        import matplotlib.pyplot as plt

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss -'+data_set+"/"+model_type)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()