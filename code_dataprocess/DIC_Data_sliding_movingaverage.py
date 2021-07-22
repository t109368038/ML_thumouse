import sparse
import  numpy as np

def sliding_window_process(inputdata, time_step):
    # datainput shape --> (data_len, 30)
    data_len = len(inputdata)
    sliding_len = data_len-(time_step-1)
    out_arr = np.zeros([sliding_len, time_step, 20])

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


def build_dictionary(voxel_input_data):
    dic_index = {}
    cc = 0
    print(voxel_input_data.shape)
    out_list = np.zeros([960,20])   # build the empty out_list array
    for data_frame in range(len(voxel_input_data)):
        tmp_voxel = voxel_input_data[data_frame]  # load current process frame
        tmp_list = []
        for i in range(25):
            for j in range(25):
                for k in range(25):
                    # dic_index[cc] = [i,j,k]
                    if tmp_voxel[0,i,j,k] == True:  # if current index [x,y,z](i,j,k) have
                        tmp_list.append(cc)
                    cc += 1
        cc = 0
        tmp_list += [0]*(20-len(tmp_list))  # output array zeros padding
        out_list[data_frame] = tmp_list
    print(' Dic_process finish')
    return  out_list  # shape is [960,20]

if __name__ == '__main__' :

#     # Gesture = ["circle"]
#     dataset =["transfer_0", "transfer_0dot05", "transfer_0dot2", "transfer_0dot5" ,"transfer_0dot8","transfer_0dot95", "transfer_1"]
    dataset =["transfer_1"]
    Gesture = ["circle", "eight", "rectangle", "up", "down", "left", "right"]
    PATH = "D:/thumouse_training_data/"
    # dir_name = 'transfer_0dot5'
for dir_name in dataset:
    head_path = PATH + dir_name+'/'
    save_path = head_path + "sliding_data/"
    name = '_scr_'+ dir_name
    time = 0
    for i in Gesture:
        for j in range(2, 4):
            print("====={}=====".format(dir_name))

            path = head_path + i + "/time" + str(j) + "/"
            out_cam_p = np.load(path + 'out_cam_p' + name + '.npy', allow_pickle=True)
            radar = np.load(path + 'out_radar_p' + name + '.npy', allow_pickle=True)

            radar = radar[20:980] # for moving average mid out
            out_cam_p = out_cam_p[20:980]

            print("-------------------{}---------------------".format("/"+i + "/time" + str(j) + "/"))
            dict_out = build_dictionary(radar)

            sliding_out = sliding_window_process(dict_out, time_step=3)
            # outcamp = sliding_window_process_camera(out_cam_p[:,:,8], time_step=3)

            np.save(save_path+ i +"_time" + str(j) + "_radar_dict.npy", sliding_out)
#             np.save(save_path+ i +"_time" + str(j) + "_cam.npy", outcamp)


