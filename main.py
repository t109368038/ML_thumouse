import datetime
import pickle
import os
import numpy as np
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.engine.saving import load_model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, TimeDistributed, LSTM, Dropout, Dense, BatchNormalization, \
    LeakyReLU
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import Adam
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend

# from learn.classes import thumouseDataGen

is_use_pre_train = False
epochs = 50000
timesteps = 4

date = 121319

## Generators
X = []
Y = []

Gesture = ["circle", "eight", "rectangle", "up", "down", "left", "right"]
head_path = 'C:/Users/user/Desktop/thmouse_training_data/'
for i in Gesture:
    for j in range(1, 4):
        tmp_path = head_path + i + "/time" + str(j) + "/"

        cam_voxel = np.load(tmp_path + "out_cam_p.npy", allow_pickle=True)
        radar_voxel = np.load(tmp_path + "out_radar_p.npy", allow_pickle=True)
        X.append(radar_voxel[:980])
        Y.append(cam_voxel[:980,:,8])
        print("cam_voxl: {} ".format(np.shape(cam_voxel)))
        print("radar_voxl: {} ".format(np.shape(radar_voxel)))


X = np.reshape(X,[20580,1,25,25,25]) # 980*3*7  frame/times/getsture
Y = np.reshape(Y,[-1,3])
print('Load Finished!')
X = np.asarray(X)
Y = np.asarray(Y)



print('Splitting test-train...')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=3, shuffle=True)

print(np.shape(X_train))
print(np.shape(Y_train))
X_train = np.reshape(X_train,[-1,timesteps,1,25,25,25])
X_test = np.reshape(X_test,[-1,timesteps,1,25,25,25])
Y_train = np.reshape(Y_train,[-1,timesteps,3])
Y_test = np.reshape(Y_test,[-1,timesteps,3])

print(np.shape(X_train))
print(np.shape(Y_train))
if not is_use_pre_train:
    print('Building Model...')
    # CRNN version ###############################################
    model = Sequential()
    model.add(
        TimeDistributed(
            Conv3D(filters=8, kernel_size=(3, 3, 3), data_format='channels_first', input_shape=(1, 25, 25, 25),
                   kernel_regularizer=l2(0.0005), kernel_initializer='random_uniform'), input_shape=(timesteps, 1, 25, 25, 25)))
    # classifier.add(TimeDistributed(LeakyReLU(alpha=0.1)))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(TimeDistributed(
        Conv3D(filters=8, kernel_size=(3, 3, 3), data_format='channels_first', kernel_regularizer=l2(0.0005))))
    # classifier.add(TimeDistributed(LeakyReLU(alpha=0.1)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling3D(pool_size=(2, 2, 2))))
    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(units=32, return_sequences=True, kernel_initializer='random_uniform'))
    model.add(TimeDistributed(Dropout(rate=0.4)))

    model.add(LSTM(units=32, return_sequences=True, kernel_initializer='random_uniform'))
    model.add(TimeDistributed(Dropout(rate=0.4)))

    model.add(LSTM(units=32, return_sequences=True, kernel_initializer='random_uniform'))
    model.add(TimeDistributed(Dropout(rate=0.4)))

    model.add(TimeDistributed(Dense(units=256)))
    model.add(TimeDistributed(LeakyReLU(alpha=0.1)))
    model.add(TimeDistributed(Dropout(0.5)))

    model.add(TimeDistributed(Dense(units=3)))
    # model.add(LSTM(units=32, return_sequences=True, kernel_initializer='random_uniform'))
    # model.add(Dropout(rate=0.4))
    #
    # model.add(LSTM(units=32, return_sequences=True, kernel_initializer='random_uniform'))
    # model.add(Dropout(rate=0.4))
    #
    # model.add(LSTM(units=32, return_sequences=False, kernel_initializer='random_uniform'))
    # model.add(Dropout(rate=0.4))
    #
    # model.add(Dense(units=256))
    # model.add(LeakyReLU(alpha=0.1))
    # model.add(Dropout(0.5))
    #
    # model.add(Dense(units=3))
    sgd = optimizers.SGD(lr=5e-5, momentum=0.9, decay=1e-6, nesterov=True)
    adam = Adam(lr=1e-6, decay=1e-6)
    model.compile(optimizer=adam, loss='mean_squared_error')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)# 當val_loss最小50Epoch 都不變
mc = ModelCheckpoint(
    'D:/pythonProject/ML_thumouse/' + str(datetime.datetime.now()).replace(':', '-').replace(
        ' ', '_') + '.h5',
    monitor='val_loss', mode='min', verbose=1, save_best_only=True)
print(model.summary())
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                    batch_size=12, epochs=epochs,  callbacks=[es, mc])


import matplotlib.pyplot as plt

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()