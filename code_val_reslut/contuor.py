import xlsxwriter
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tensorflow import keras


workbook = xlsxwriter.Workbook('write_data.xlsx')
worksheet = workbook.add_worksheet()

def show_error(x, y, z):
    from matplotlib import gridspec
    mark = ","
    size = 1
    fig = plt.figure()
    spec = gridspec.GridSpec(ncols=3, nrows=1)
    ax0 = fig.add_subplot(spec[0])
    x = np.array(x)
    x = x * -1
    axis_len = 8
    print(np.shape(x))
    ax0.set_title('X-Y plane', color='black')
    ax0.scatter(x, y,s=size, marker=mark)
    ax0.set_xlim(-1*axis_len, axis_len)
    ax0.set_ylim(-1*axis_len, axis_len)
    ax0.set_xlabel("x(cm)")
    ax0.set_ylabel("y(cm)")
    ax0.set_aspect(1)
    ax0.legend()

    ax1 = fig.add_subplot(spec[1])
    ax1.scatter(x, z, s=size, marker=mark)
    ax1.set_xlim(-1*axis_len, axis_len)
    ax1.set_ylim(-1*axis_len, axis_len)
    ax1.set_xlabel("x(cm)")
    ax1.set_ylabel("z(cm)")
    ax1.set_aspect(1)
    ax1.set_title('X-Z plane', color='black')
    ax1.legend()

    ax2 = fig.add_subplot(spec[2])
    ax2.scatter(y, z, s=size, marker=mark,)
    ax2.set_xlim(-1*axis_len, axis_len)
    ax2.set_ylim(-1*axis_len, axis_len)
    ax2.set_xlabel("y(cm)")
    ax2.set_ylabel("z(cm)")
    ax2.set_title('Y-Z plane', color='black')
    ax2.legend()
    ax2.set_aspect(1)
    fig.tight_layout()
    fig.suptitle(title, fontsize=20)
    plt.show()
    # plt.savefig(path+"Contour(t2+t3)timestep12.png")
    # plt.savefig(path+"Contour(t2+t3).png")
    # plt.close()

def mean_square_error( groundtruth, outresult, title):
    x = groundtruth
    y = outresult
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
    return  stdx, stdy, stdz

Gesture = ["circle", "eight", "rectangle", "up", "down", "left", "right"]
head_path = 'C:/Users/user/Desktop/thmouse_training_data/'
time = 0
for i in Gesture:
    for j in range(2, 4):
        tmp_path = head_path + i + "/time" + str(j) + "/"
        title = i + " of /time" + str(j) + " ML-timestep4(t2+t3)"
        path = tmp_path
        out_cam_p = np.load(path + 'out_cam_p.npy', allow_pickle=True)
        radar = np.load(path + 'out_radar_p_larry.npy', allow_pickle=True)
        out_radar_p = np.load(path + 'out_center_p.npy', allow_pickle=True)
        end_len =980
        out_cam_p  = out_cam_p[20:end_len,:,8]
        out_radar_p  = out_radar_p[20:end_len]
        radar = np.reshape(radar[20:end_len],[-1,20,1,25,25,25])
        # radar = np.reshape(radar[20:end_len],[-1,12,1,25,25,25])
        # radar = np.reshape(radar[20:end_len],[-1,4,1,25,25,25])
        # radar = np.reshape(radar[20:140],[-1,3,1,25,25,25])

        # x, y, z, stdx, stdy, stdz =mean_square_error(out_cam_p, out_radar_p,title)
        # show_error(out_cam_p, out_radar_p, x, y, z, stdx, stdy, stdz, title, tmp_path)

        # model = keras.models.load_model("D:\\pythonProject\\ML_thumouse\\time2_3.h5")
        # model = keras.models.load_model("D:\\pythonProject\\ML_thumouse\\timestep_12.h5")
        model = keras.models.load_model("D:\\pythonProject\\ML_thumouse\\Model\\timestep_20.h5")

        out = model.predict(radar)
        out = np.reshape(out,[-1,3])
        # out_cam_p  = out_cam_p[3::4]
        x, y, z = mean_square_error(out_cam_p, out ,title)
        show_error(x,y,z)



