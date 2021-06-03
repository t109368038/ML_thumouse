import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from tensorflow import keras
import xlsxwriter

workbook = xlsxwriter.Workbook('write_data.xlsx')
worksheet = workbook.add_worksheet()

def show_error(camera_point,predict_p,x, y, z, stdx, stdy, stdz, title, savepath):
    out_cam_p = camera_point
    out_radar_p = predict_p
    print(out_cam_p.shape)
    print(out_radar_p.shape)
    total_len = range(len(out_cam_p))

    plt.subplot(3, 1, 1)
    tmp_1 = np.mean(out_cam_p)
    tmp_2 = np.mean(out_radar_p)
    tmp  = tmp_1 - tmp_2
    out_cam_p += tmp
    plt.plot(total_len, out_cam_p[:,0], label="camera")
    plt.plot(total_len, out_radar_p[:,0], label="radar")
    plt.title('ALL frame on X axis: mse = {}cm^2, std = {}cm'.format(np.round(x, 3), np.round(stdx, 3)), color='black')
    plt.legend()

    plt.subplot(3, 1, 2)
    tmp_1 = np.mean(out_cam_p)
    tmp_2 = np.mean(out_radar_p)
    tmp  = tmp_1 - tmp_2
    plt.plot(total_len, out_cam_p[:,1], label="camera")
    plt.plot(total_len, out_radar_p[:,1], label="radar")
    plt.title('ALL frame on Y axis: mse = {}cm^2, std = {}cm'.format(np.round(y, 3), np.round(stdy, 3)), color='black')
    plt.legend()

    plt.subplot(3, 1, 3)
    tmp_1 = np.mean(out_cam_p)
    tmp_2 = np.mean(out_radar_p)
    tmp  = tmp_1 - tmp_2
    plt.plot(total_len, out_cam_p[:,2], label="camera")
    plt.plot(total_len, out_radar_p[:,2], label="radar")
    plt.legend()
    plt.title('ALL frame on Z axis: mse = {}cm^2, std = {}cm'.format(np.round(z, 3), np.round(stdz, 3)), color='black')
    plt.suptitle(title, fontsize=20)
    plt.subplots_adjust(hspace=0.8)
    # plt.show()
    # plt.savefig(path+"without_ML.png")
    plt.savefig(path+"with_ML_timestep3.png")
    plt.close()

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
    return  errorx/cc, errory/cc, errorz/cc, std_x, std_y, std_z

Gesture = ["circle", "eight", "rectangle", "up", "down", "left", "right"]
head_path = 'C:/Users/user/Desktop/thmouse_training_data/'
time = 0
for i in Gesture:
    for j in range(1, 4):
        tmp_path = head_path + i + "/time" + str(j) + "/"
        title = i + " of /time" + str(j) + " ML-timestep3"
        path = tmp_path
        out_cam_p = np.load(path + 'out_cam_p.npy', allow_pickle=True)
        radar = np.load(path + 'out_radar_p.npy', allow_pickle=True)
        out_radar_p = np.load(path + 'out_center_p.npy', allow_pickle=True)

        out_cam_p  = out_cam_p[20:140,:,8]
        out_radar_p  = out_radar_p[20:140]
        # radar = np.reshape(radar[20:140],[-1,12,1,25,25,25])
        # radar = np.reshape(radar[20:140],[-1,4,1,25,25,25])
        radar = np.reshape(radar[20:140],[-1,3,1,25,25,25])
        # x, y, z, stdx, stdy, stdz =mean_square_error(out_cam_p, out_radar_p,title)
        # show_error(out_cam_p, out_radar_p, x, y, z, stdx, stdy, stdz, title, tmp_path)

        # model = keras.models.load_model("D:\\pythonProject\\ML_thumouse\\timestep_20\\2021-05-31_15-41-51.311804.h5")
        # model = keras.models.load_model("D:\\pythonProject\\ML_thumouse\\timestep_4\\2021-06-01_02-14-15.930120.h5")
        model = keras.models.load_model("D:\\pythonProject\\ML_thumouse\\timestep_3\\2021-05-31_22-05-47.686357.h5")
        out = model.predict(radar)
        out = np.reshape(out,[-1,3])
        x, y, z, stdx, stdy, stdz = mean_square_error(out_cam_p, out,title)
        # show_error(out_cam_p, out, x, y, z, stdx, stdy, stdz, title, tmp_path)

        worksheet.write(time, 0, title)  # Writes an int
        worksheet.write(time, 1, "mse")  # Writes a float
        worksheet.write(time, 2, x)  # Writes a float
        worksheet.write(time, 3, y)  # Writes a string
        worksheet.write(time, 4, z)  # Writes None
        worksheet.write(time, 5, stdx)  # Writes None
        worksheet.write(time, 6, stdy)  # Writes None
        worksheet.write(time, 7, stdz)  # Writes None
        time= time+1


workbook.close()
