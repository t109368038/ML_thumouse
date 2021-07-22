import numpy as np
import xlsxwriter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tensorflow import keras
from matplotlib import gridspec

workbook = xlsxwriter.Workbook('write_data.xlsx')
worksheet = workbook.add_worksheet()

def show_trace(cam,out,title):

    # print(np.shape(cam))
    # print(np.shape(out))
    cam = cam/0.015
    out = out/0.015
    mark = ","
    size = 1
    fig = plt.figure()
    spec = gridspec.GridSpec(ncols=2, nrows=1)
    ax0 = fig.add_subplot(spec[0])

    ax0.set_title('X-Y plane', color='black')
    ax0.scatter(cam[:, 0], cam[:, 1],s=size, marker=mark)
    ax0.plot(cam[:, 0], cam[:, 1])
    ax0.scatter(out[:, 0], out[:, 1], s=size, marker=mark)
    ax0.plot(out[:, 0], out[:, 1])
    # ax0.set_xlim(-10, 10)
    # ax0.set_ylim(-10, 10)
    ax0.set_xlabel("x(cm)")
    ax0.set_ylabel("y(cm)")
    ax0.set_aspect(1)
    ax0.legend()

    ax1 = fig.add_subplot(spec[1])
    ax1.scatter(cam[:, 0], cam[:, 2],s=size, marker=mark)
    ax1.plot(cam[:, 0], cam[:, 2])
    ax1.scatter(out[:, 0], out[:, 2],s=size, marker=mark)
    ax1.plot(out[:, 0], out[:, 2])
    # ax1.set_xlim(-10, 10)
    # ax1.set_ylim(-10, 10)
    ax1.set_xlabel("x(cm)")
    ax1.set_ylabel("z(cm)")
    ax1.set_aspect(1)
    ax1.set_title('X-Z plane', color='black')
    ax1.legend()
    plt.suptitle(title)
    fig.tight_layout()
    plt.show()
    # plt.savefig(path+"Trace(t2+t3)timestep12.png")
    # plt.close()

def mean_square_error( x, y, title):
    cc = 0
    errorx = 0
    errory = 0
    errorz = 0
    stdx = []
    stdy = []
    stdz = []
    for i in range(len(x)):
        if y[i,0]== 0 and y[i,1]== 0 and y[i,2]== 0:
            pass
        else:
            errorx += (((x[i, 0] - y[i, 0])/0.015)**2)
            errory += (((x[i, 1] - y[i, 1])/0.015)**2)
            errorz += (((x[i, 2] - y[i, 2])/0.015)**2)
            stdx.append((x[i, 0]- y[i, 0])/0.015)
            stdy.append((x[i, 1]- y[i, 1])/0.015)
            stdz.append((x[i, 2]- y[i, 2])/0.015)
            cc +=1
    # std_x = np.sqrt(sum(stdx - np.mean(stdx)))
    # std_y = np.sqrt(sum(stdy - np.mean(stdy)))
    # std_z = np.sqrt(sum(stdz - np.mean(stdz)))
    std_x = np.std(stdx)
    std_y = np.std(stdy)
    std_z = np.std(stdz)
    print("\n-----  {} -----".format(title))
    print("std x:{}  y:{}  z:{}".format(np.round(std_x,3),np.round(std_y,3),np.round(std_z,3)))
    print("MSE error x:{}   y:{}   z:{}".format(np.round(errorx/cc,3),np.round(errory/cc,3),np.round(errorz/cc,3)))

    errorxy = (errorx / cc +  errory / cc)/2
    stdxy = (std_x + std_y )/2
    errorxz = (errorx / cc  + errorz / cc) / 2
    stdxz = (std_x + std_z) / 2

    print("XY mse:{}    std:{}" .format(np.round(errorxy,3),np.round(stdxy,3)))
    print("XZ mse:{}    std:{}".format(np.round(errorxz,3),np.round(stdxz,3)))

    return  errorx/cc, errory/cc, errorz/cc, std_x, std_y, std_z

Gesture = ["circle", "eight", "rectangle", "up", "down", "left", "right"]
# Gesture = ["right"]
timesteps = 3
time = 0
name ='transfer_0dot5'
model_name = "sparse_xyzxyz_" + name + "_2lstm"
# model_name= "sparse_xyzxyz_"+name+"_1lstm"
# model_name= "sparse_xxyyzz_"+name+"_2lstm"
# model_name= "sparse_xxyyzz_"+name+"_1lstm"
for i in Gesture:
    for j in range(2, 4):
        head_path = "D:/thumouse_training_data/model/"
        path = 'D:/thumouse_training_data/'+ name +'/sliding_data/'
        # path = 'C:/Users/user/Desktop/thmouse_training_data/with_Larry_scr/larry_static/timestep'+str(timesteps)+'/'
        # path = 'C:/Users/user/Desktop/thmouse_training_data/with_Larry_scr/normal_static/timestep'+str(timesteps)+'/'
        out_cam_p = np.load(path + i + "_time" + str(j) + "_cam.npy", allow_pickle=True)
        out_radar_p = np.load(path + i + "_time" + str(j) + "_radar_xyzxyz.npy", allow_pickle=True)
        # out_radar_p = np.load(path + i + "_time" + str(j) + "_radar_xxyyzz.npy", allow_pickle=True)

        if timesteps == 3:
            out_radar_p = out_radar_p[100:118]
            out_cam_p = out_cam_p[100:118,-1, :]
        elif timesteps == 12:
            out_radar_p = out_radar_p[97:109]
            out_cam_p = out_cam_p[97:109,-1, :]
        print(np.shape(out_cam_p))
        print(np.shape(out_radar_p))

        model = keras.models.load_model(head_path+model_name+".h5")


        out = model.predict(out_radar_p)
        out = out[:,-1,:]
        print(np.shape(out))


        title = str(i) + "/times"+str(j)+'/'+model_name
        show_trace(out_cam_p,out,title)
        mean_square_error(out_cam_p,out,"")


