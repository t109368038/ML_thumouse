import  numpy as np

def sliding_window_process(inputdata, time_step):
    # datainput shape --> (data_len, 1, 25, 25, 25)
    data_len = len(inputdata)
    sliding_len = data_len-(time_step-1)
    out_arr = np.zeros([sliding_len, time_step, 1, 25, 25, 25])

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
    head_path = 'C:/Users/user/Desktop/thmouse_training_data/'
    save_path = 'C:/Users/user/Desktop/thmouse_training_data/with_Larry_scr/'
    time = 0
    for i in Gesture:
        for j in range(2, 4):
        # i = 0
        # j = 2
            tmp_path = head_path + i + "/time" + str(j) + "/"
            path = tmp_path
            out_cam_p = np.load(path + 'out_cam_p.npy', allow_pickle=True)
            radar = np.load(path + 'out_radar_p.npy', allow_pickle=True)
            out_radar_p = np.load(path + 'out_center_p.npy', allow_pickle=True)

            radar = radar[20:980]
            out_cam_p = out_cam_p[20:980]
            outdata = sliding_window_process(radar, 3)
            outcamp = sliding_window_process_camera(out_cam_p[:,:,8], 3)
            np.save(save_path+ i +"_time" + str(j) + "_radar.npy", outdata)
            np.save(save_path+ i +"_time" + str(j) + "_cam.npy", outcamp)



