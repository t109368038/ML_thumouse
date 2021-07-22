import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from tensorflow import keras
cam = np.load("out_cam_p.npy", allow_pickle=True)
radar = np.load("out_radar_p.npy", allow_pickle=True)
radar = radar[:180]
cam = cam[:180,:,8]
X = cam[:, 0]
Y = cam[:, 1]
Z = cam[:, 2]
timestep = 12
radar = np.reshape(radar, [-1, timestep, 1, 25, 25, 25])

show_only_p = False
model = keras.models.load_model("D:\\pythonProject\\ML_thumouse\\timestep_20\\2021-05-31_15-41-51.311804.h5")
out = model.predict(radar)
out = np.reshape(out,[-1,3])
fig = plt.figure()
ax = fig.gca(projection='3d')
for i in range(180):
    if show_only_p == True:
        tX = X[0+i]
        tY = Y[0+i]
        tZ = Z[0+i]
        ansX = out[0 + i, 0]
        ansY = out[0 + i, 1]
        ansZ = out[0 + i, 2]
    else:
        tX = X[0+i:6+i]
        tY = Y[0+i:6+i]
        tZ = Z[0+i:6+i]
        ansX = out[0+i:6+i, 0]
        ansY = out[0+i:6+i, 1]
        ansZ = out[0+i:6+i, 2]
    ax.scatter(tX, tY, tZ, cmap='jet', label='camera points')
    ax.scatter(ansX, ansY, ansZ, cmap='jet', label='predict Points')

    ax.scatter(0, 0, 0, cmap='Reds', marker='^', label='radar center')
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(0, 0.3)
    ax.set_zlim(-0.2, 0.2)
    ax.view_init(azim=76, elev=4)
    # ax.view_init(azim=4, elev=4)
    ax.legend()
    plt.draw()
    plt.pause(.05)
    # plt.pause(100)
    plt.cla()