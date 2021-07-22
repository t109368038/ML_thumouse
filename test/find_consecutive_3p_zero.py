import numpy as np
import xlsxwriter

def excel_init(name):
    time = 0
    workbook = xlsxwriter.Workbook(name + '.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write(time, 0, "Data")  # Writes an int
    worksheet.write(time, 1, "time")  # Writes a float
    return workbook, worksheet, time

def find_consecutive_3p_zero(index):
    count = 0
    for i in range(len(index)-2):
        t1 = index[i]
        t2 = index[i+1]
        t3 = index[i+2]
        if (t2-t1) !=1 :
            continue
        else:
            if (t3-t2) !=1 :
                continue
            else:
               # print(t1,t2,t3)
               count+=1
    return count
Gesture = ["circle", "eight", "rectangle", "up", "down", "left", "right"]
# Gesture = ["circle"]
head_path = 'C:/Users/user/Desktop/thmouse_training_data/'
workbook,worksheet ,time = excel_init("show3point_zero_number")
for i in Gesture:
    for j in range(2, 4):
        tmp_path = head_path + i + "/time" + str(j) + "/"
        title = i + " of /time" + str(j)
        path = tmp_path
        out_cam_p = np.load(path + 'out_cam_p.npy', allow_pickle=True)
        radar = np.load(path + 'out_radar_p.npy', allow_pickle=True)
        print(radar.shape)
        index = []
        for k in range(len(radar)):
            # print(f"Number of Zeroes in Array -->{radar[k][np.where(radar[k] != 0)].size}/{radar[k][np.where(radar[k] == 0)].size}")
            if radar[k][np.where(radar[k] != 0)].size == 0:
              index.append(k)

        cc = find_consecutive_3p_zero(index)
        time += 1
        worksheet.write(time, 0, i+ "/time" + str(j) + "/sliding data/ countinuse 3p zeros")  # Writes an int
        worksheet.write(time, 1, cc)  # Writes a float
        print(i + "/time" + str(j) + "/sliding data/ countinuse 3p zeros : {}".format(cc))

workbook.close()