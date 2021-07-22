import numpy as np
import xlsxwriter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tensorflow import keras
from matplotlib import gridspec

workbook = xlsxwriter.Workbook('write_data.xlsx')
worksheet = workbook.add_worksheet()

def show_trace(cam,out,title):

    print(np.shape(cam))
    print(np.shape(out))
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
    fig.tight_layout()
    plt.suptitle(title)
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

# Gesture = ["circle"]
# Gesture = ["circle", "eight", "rectangle", "up", "down", "left", "right"]
Gesture = ["circle", "down"]
head_path = 'C:/Users/user/Desktop/thmouse_training_data/'
time = 0
for i in Gesture:
    for j in range(2, 3):
        tmp_path = head_path + i + "/time" + str(j) + "/"
        title = i + " of /time" + str(j) + " /light_nonesliding_larrysrc_timestep12"
        path = tmp_path
        out_cam_p = np.load(path + 'out_cam_p.npy', allow_pickle=True)
        radar = np.load(path + 'out_radar_p_larry.npy', allow_pickle=True)
        # radar = np.load(path + 'out_radar_p.npy', allow_pickle=True)
        out_radar_p = np.load(path + 'out_center_p.npy', allow_pickle=True)
        start_len = 128
        end_len = 140
        out_cam_p  = out_cam_p[start_len:end_len,:,8]
        out_radar_p  = out_radar_p[start_len:end_len]
        # radar = np.reshape(radar[start_len:end_len],[-1,20,1,25,25,25])
        # radar = np.reshape(radar[start_len:end_len],[-1,12,1,25,25,25])
        # radar = np.reshape(radar[start_len:end_len],[-1,4,1,25,25,25])
        radar = np.reshape(radar[start_len:end_len],[-1,3,1,25,25,25])
        # radar = np.reshape(radar[20:140],[-1,3,1,25,25,25])
        # x, y, z, stdx, stdy, stdz =mean_square_error(out_cam_p, out_radar_p,title)
        # show_error(out_cam_p, out_radar_p, x, y, z, stdx, stdy, stdz, title, tmp_path)

        model = keras.models.load_model("D:\\pythonProject\\ML_thumouse\\New_model_adam\\light_nonesliding_larrysrc_timestep12.h5")

        out = model.predict(radar)
        out = np.reshape(out,[-1,3])
        # tmp_outcamp =  out_cam_p[3::4]
        show_trace(out_cam_p,out,title)
        mean_square_error(out_cam_p,out,"")


##　single test


# tmp_path = 'C:/Users/user/Desktop/thmouse_training_data/'
#
# path = tmp_path
# out_cam_p = np.load(path + 'out_cam_p.npy', allow_pickle=True)
# radar = np.load(path + 'out_radar_p.npy', allow_pickle=True)
# out_radar_p = np.load(path + 'out_center_p.npy', allow_pickle=True)
# start_len = 0
# end_len =40
# out_cam_p  = out_cam_p[start_len:end_len,:,8]
# out_radar_p  = out_radar_p[start_len:end_len]
# # radar = np.reshape(radar[20:140],[-1,12,1,25,25,25])
# radar = np.reshape(radar[start_len:end_len],[-1,4,1,25,25,25])
# # radar = np.reshape(radar[20:140],[-1,3,1,25,25,25])
# # x, y, z, stdx, stdy, stdz =mean_square_error(out_cam_p, out_radar_p,title)
# # show_error(out_cam_p, out_radar_p, x, y, z, stdx, stdy, stdz, title, tmp_path)
#
# model = keras.models.load_model("D:\\pythonProject\\ML_thumouse\\t2t3新方法.h5")
# out = model.predict(radar)
# out = np.reshape(out,[-1,3])
#
# tmp_outcamp =  out_cam_p[4::4]
# show_trace(tmp_outcamp,out)