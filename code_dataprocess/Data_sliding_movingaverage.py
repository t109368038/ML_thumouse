import  numpy as np

def sliding_window_process(inputdata, time_step,voxel_size):
    # datainput shape --> (data_len, 1, 25, 25, 25)
    data_len = len(inputdata)
    sliding_len = data_len-(time_step-1)
    out_arr = np.zeros([sliding_len, time_step, 1, voxel_size, voxel_size, voxel_size])

    for i in range(sliding_len):
        tmp = []
        for j in range(time_step):
            tmp.append(inputdata[i+j])
            out_arr[i,j] = tmp[j]
    print(np.shape(out_arr))
    return  out_arr

def sliding_window_process_camera(inputdata, time_step):
    # datainput shape --> (data_len, 3, 21)

    data_len = len(inputdata)
    sliding_len = data_len-(time_step-1)
    out_arr = np.zeros([sliding_len, time_step, 3])
    for i in range(sliding_len):
        tmp = []
        for j in range(time_step):
            tmp.append(inputdata[i+j])
            out_arr[i,j] = tmp[j]
    print(np.shape(out_arr))
    return  out_arr


if __name__ == '__main__' :

    Gesture = ["circle", "eight", "rectangle", "up", "down", "left", "right"]
    path1  = "D:/thumouse_training_data/moving_average_out_last/"
    path2  = "D:/thumouse_training_data/moving_average_out_mid/"
    path3  = "D:/thumouse_training_data/moving_average_out_last_range_test/"
    path4  = "D:/thumouse_training_data/moving_average_out_mid_range_test/"
    path5  = "D:/thumouse_training_data/transfer_0dot8/"
    path6  = "D:/thumouse_training_data/moving_average3_out_mid_range_test/"
    path7  = "D:/thumouse_training_data/moving_average3_out_last_range_test/"
    path8  = "D:/thumouse_training_data/moving_average3_out_mid/"
    path9  = "D:/thumouse_training_data/transfer_0dot5/"
    # PATH = "D:/thumouse_training_data/"
    PATH = "D:/thumouse_training_data_new/"
    dir_name = 'transfer_0dot5'
    head_path = PATH + dir_name+'/'
    save_path = head_path + "sliding_data/"
    # name = '_scr_moving_average_out_last'
    # name = '_scr_moving_average_out_mid'
    name = '_scr_'+ dir_name
    time = 0
    for i in Gesture:
        for j in range(2, 4):

            path = head_path + i + "/time" + str(j) + "/"
            out_cam_p = np.load(path + 'out_cam_p' + name + '.npy', allow_pickle=True)
            radar = np.load(path + 'out_radar_p' + name + '.npy', allow_pickle=True)

            radar = radar[100:1180]
            out_cam_p = out_cam_p[100:1180]
            # print(radar.shape)
            # print(out_cam_p.shape)
            #
            outdata = sliding_window_process(radar, 3, 32)
            outcamp = sliding_window_process_camera(out_cam_p[:,:,8], 3)

            np.save(save_path+ i +"_time" + str(j) + "_radar.npy", outdata)
            np.save(save_path+ i +"_time" + str(j) + "_cam.npy", outcamp)



