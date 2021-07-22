import sparse
import  numpy as np

def sliding_window_process(inputdata, time_step):
    # datainput shape --> (data_len, 30)
    data_len = len(inputdata)
    sliding_len = data_len-(time_step-1)
    out_arr = np.zeros([sliding_len, time_step, 60])

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


def voxel2zigzac(voxel_data ,zigzac_index):
    # print(zigzac_index.shape)
    out_data = np.zeros([len(voxel_data),60])
    for i in range(len(voxel_data)):
        tmp_data = voxel_data[i]
        tmp_out = np.zeros([1,60])
        index = 0
        for pos in zigzac_index:
            is_voxel = tmp_data[0, pos[0], pos[1], pos[2]]
            if is_voxel == 1:
                # print(pos[0], pos[1], pos[2])
                tmp_out[0, index*3] = pos[0]
                tmp_out[0, index*3+1] = pos[1]
                tmp_out[0, index*3+2] = pos[2]
                index += 1
        out_data[i] = tmp_out[0]
    print("zigzac_process finish")
    return  out_data


if __name__ == '__main__' :
    zigzag_index = np.load("25by25zigzag_index.npy", allow_pickle=True)  # load zigzag
    zigzag_index -= 1
    # Gesture = ["circle"]
    Gesture = ["circle", "eight", "rectangle", "up", "down", "left", "right"]
    PATH = "D:/thumouse_training_data/"
    dir_name = 'transfer_0dot5'
    head_path = PATH + dir_name+'/'
    save_path = head_path + "zigzag_data/"
    name = '_scr_'+dir_name
    time = 0
    for i in Gesture:
        # for j in range(2, 3):
        for j in range(2, 4):
            path = head_path + i + "/time" + str(j) + "/"
            out_cam_p = np.load(path + 'out_cam_p' + name + '.npy', allow_pickle=True)
            radar = np.load(path + 'out_radar_p' + name + '.npy', allow_pickle=True)

            radar = radar[20:980]
            out_cam_p = out_cam_p[20:980]
            print(i + "/time" + str(j) + "/ zigzac process start")
            zigzag_out = voxel2zigzac(radar, zigzag_index)
            outdata= sliding_window_process(zigzag_out, time_step=3)
            outcamp = sliding_window_process_camera(out_cam_p[:,:,8], time_step=3)
            np.save(save_path + i + "_time" + str(j) + "_radar_zigzag.npy", outdata)
            np.save(save_path + i + "_time" + str(j) + "_cam.npy", outcamp)



