import numpy as np

look = "2"

if look == "1":
    from tensorflow import keras

    # model = keras.models.load_model("D:\\pythonProject\\ML_thumouse\\Model\\single_direction.h5")
    model = keras.models.load_model("D:\\pythonProject\\ML_thumouse\\Model\\timestep_12.h5")
    weights = model.get_weights() # returs a n umpy list of weights
    for i in range(len(weights)):
        print(f"Number of Zeroes in Array --> {weights[i][np.where(weights[i] == 0)].size}")

elif look == "2":
    import xlsxwriter

    workbook = xlsxwriter.Workbook('count_data.xlsx')
    worksheet = workbook.add_worksheet()
    Gesture = ["circle", "eight", "rectangle", "up", "down", "left", "right"]
    head_path = 'C:/Users/user/Desktop/thmouse_training_data/'
    time = 0
    for i in Gesture:
        for j in range(2, 4):
            tmp_path = head_path + i + "/time" + str(j) + "/"
            title = i + " of /time" + str(j) + " ML-timestep12(t2+t3)"
            path = tmp_path
            out_cam_p = np.load(path + 'out_cam_p.npy', allow_pickle=True)
            radar_larry_src = np.load(path + 'out_radar_p_larry.npy', allow_pickle=True)
            radar_src = np.load(path + 'out_radar_p.npy', allow_pickle=True)
            out_radar_p = np.load(path + 'out_center_p.npy', allow_pickle=True)
            zeropd = 0
            zeropd1 = 0

            # print("Gesture: {} times: {} ".format( i, j))
            count_dict_larry=  {"0":0}
            count_dict_origin=  {"0":0}
            for k in range(len(radar_larry_src)):
                #---------------------------------------------------
                #        moving average - static clutter removal
                #---------------------------------------------------
                if radar_larry_src[k][np.where(radar_larry_src[k] != 0)].size ==0:
                    count_dict_larry["0"] += 1
                    zeropd += 1
                else:
                    tmp_c = str(radar_larry_src[k][np.where(radar_larry_src[k] != 0)].size)
                    if tmp_c in count_dict_larry:
                        count_dict_larry[tmp_c] += 1
                    else:
                        count_dict_larry[tmp_c] = 1
                #---------------------------------------------------
                #           normal static clutter removal
                #---------------------------------------------------
                if radar_src[k][np.where(radar_src[k] != 0)].size ==0:
                    count_dict_origin["0"] += 1
                    zeropd1 += 1
                else:
                    tmp_c = str(radar_src[k][np.where(radar_src[k] != 0)].size)
                    if tmp_c in count_dict_origin:
                        count_dict_origin[tmp_c] += 1
                    else:
                        count_dict_origin[tmp_c] = 1

            print("\n\n ======{}======".format(str(i) +"_time"+ str(j)))
            print("zero of pd numbs larry :{}/{} ".format(zeropd,len(radar_larry_src)))
            tmp = 0
            worksheet.write(time, tmp, str(i) +"_time"+ str(j))  # Writes an int
            time +=1
            worksheet.write(time, tmp, "出現點雲數量")  # Writes an int
            worksheet.write(time + 1, tmp, "次數")  # Writes a float
            for k,v in sorted(count_dict_larry.items()):
                tmp += 1
                worksheet.write(time  , tmp, k)  # Writes an int
                worksheet.write(time+1, tmp, v)  # Writes a float
            time = time  +2

            print("zero of pd numbs origin:{}/{} ".format(zeropd1,len(radar_larry_src)))
            tmp = 0
            worksheet.write(time, tmp, "出現點雲數量")  # Writes an int
            worksheet.write(time + 1, tmp, "次數")  # Writes a float
            for k, v in sorted(count_dict_origin.items()):
                tmp += 1
                worksheet.write(time  , tmp, k)  # Writes an int
                worksheet.write(time+1, tmp, v)  # Writes a float
            time = time  +2

workbook.close()
