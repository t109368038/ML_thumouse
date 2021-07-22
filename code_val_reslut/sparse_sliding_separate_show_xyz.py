import numpy as np
import xlsxwriter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tensorflow import keras



def show_error(camera_point,predict_p,x, y, z, stdx, stdy, stdz, title):
    out_cam_p = camera_point
    out_radar_p = predict_p
    # print(out_cam_p.shape)
    # print(out_radar_p.shape)
    total_len = range(len(out_cam_p))

    plt.subplot(3, 1, 1)
    tmp_1 = np.mean(out_cam_p)
    tmp_2 = np.mean(out_radar_p)
    tmp  = tmp_1 - tmp_2
    out_cam_p += tmp
    cm = 0.015
    plt.plot(total_len, out_cam_p[:,0]/cm, label="camera")
    plt.plot(total_len, out_radar_p[:,0]/cm, label="radar")
    plt.ylabel("cm")
    plt.title('ALL frame on X axis: mse = {}cm^2, std = {}cm'.format(np.round(x, 3), np.round(stdx, 3)), color='black')
    plt.legend()

    plt.subplot(3, 1, 2)
    tmp_1 = np.mean(out_cam_p)
    tmp_2 = np.mean(out_radar_p)
    tmp  = tmp_1 - tmp_2
    plt.plot(total_len, out_cam_p[:,1]/cm, label="camera")
    plt.plot(total_len, out_radar_p[:,1]/cm, label="radar")
    plt.ylabel("cm")
    plt.title('ALL frame on Y axis: mse = {}cm^2, std = {}cm'.format(np.round(y, 3), np.round(stdy, 3)), color='black')
    plt.legend()

    plt.subplot(3, 1, 3)
    tmp_1 = np.mean(out_cam_p)
    tmp_2 = np.mean(out_radar_p)
    tmp  = tmp_1 - tmp_2
    plt.plot(total_len, out_cam_p[:,2]/cm, label="camera")
    plt.plot(total_len, out_radar_p[:,2]/cm, label="radar")
    plt.ylabel("cm")
    plt.legend()
    plt.title('ALL frame on Z axis: mse = {}cm^2, std = {}cm'.format(np.round(z, 3), np.round(stdz, 3)), color='black')
    plt.suptitle(title, fontsize=20)
    plt.subplots_adjust(hspace=0.8)
    plt.show()
    # plt.savefig(path+"without_ML.png")
    # plt.savefig(path+"with_ML_timestep20(t2+t3).png")
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
    return  errorx/cc, errory/cc, errorz/cc, std_x, std_y, std_z

def open_xlsx(name):
    workbook = xlsxwriter.Workbook(name + '.xlsx')
    worksheet = workbook.add_worksheet()
    time = 0
    worksheet.write(time, 0, "gesture/time")  # Writes an int
    worksheet.write(time, 1, "")  # Writes a float
    worksheet.write(time, 2, "msex")  # Writes a float
    worksheet.write(time, 3, "msey")  # Writes a string
    worksheet.write(time, 4, "msez")  # Writes None
    worksheet.write(time, 5, "stdx")  # Writes None
    worksheet.write(time, 6, "stdy")  # Writes None
    worksheet.write(time, 7, "stdz")  # Writes None
    worksheet.write(time, 8, "msexy")  # Writes None
    worksheet.write(time, 9, "msexz")  # Writes None
    worksheet.write(time, 10, "msexyz")  # Writes None
    worksheet.write(time, 11, "stdxyz")  # Writes None
    return workbook, worksheet, time

name ='transfer_0dot5'
# name ='transfer_0dot95'

Gesture = ["circle", "eight", "rectangle", "up", "down", "left", "right"]
# Gesture = ["circle"]
timesteps = 3

# model_name = "sparse_xyzxyz_" + name + "_2lstm"
# model_name= "sparse_xyzxyz_"+name+"_1lstm"
# model_name= "sparse_xxyyzz_"+name+"_2lstm"
# model_name= "sparse_xxyyzz_"+name+"_1lstm"
model_name= "epoch1773_transfer_0dot5_xxyyzz_2lstm"

workbook, worksheet, time = open_xlsx(model_name)
for i in Gesture:
    for j in range(2, 4):
        path = 'D:/thumouse_training_data/'+ name +'/sliding_data/'
        head_path = "D:/thumouse_training_data/model/"
        out_cam_p = np.load(path + i + "_time" + str(j) + "_cam.npy", allow_pickle=True)
        # out_radar_p = np.load(path + i + "_time" + str(j) + "_radar_xyzxyz.npy", allow_pickle=True)
        out_radar_p = np.load(path + i + "_time" + str(j) + "_radar_xxyyzz.npy", allow_pickle=True)
        print(np.shape(out_cam_p))
        print(np.shape(out_radar_p))
        if timesteps == 3:
            out_radar_p = out_radar_p[:118]
            out_cam_p = out_cam_p[:118 , -1, :]
        elif timesteps == 12:
            out_radar_p = out_radar_p[:109]
            out_cam_p = out_cam_p[:109,-1, :]

        print("out_cam_p: {}".format(np.shape(out_cam_p)))
        print("out_radar_p: {}".format(np.shape(out_radar_p)))


        model = keras.models.load_model(head_path+model_name+".h5")
        out = model.predict(out_radar_p)
        out = out[:,-1,:]
        print("out: {}".format(np.shape(out)))
        title = str(i) + "/times"+str(j)
        x, y, z, stdx, stdy, stdz = mean_square_error(out_cam_p, out, title)
        show_error(out_cam_p, out, x, y, z, stdx, stdy, stdz, title)
        time= time+1
        worksheet.write(time, 0, title)  # Writes an int
        worksheet.write(time, 1, "---")  # Writes a float
        worksheet.write(time, 2, x)  # Writes a float
        worksheet.write(time, 3, y)  # Writes a string
        worksheet.write(time, 4, z)  # Writes None
        worksheet.write(time, 5, stdx)  # Writes None
        worksheet.write(time, 6, stdy)  # Writes None
        worksheet.write(time, 7, stdz)  # Writes None
        worksheet.write(time, 8,  (x+y)/2)  # msexy
        worksheet.write(time, 9,  (x+z)/2)  # Writes None"msexz"
        worksheet.write(time, 10, (x+y+z)/3)  # Writes None"msexyz"
        worksheet.write(time, 11, (stdx+stdy+stdz)/3)  # Writes None"stdxyz"
workbook.close()