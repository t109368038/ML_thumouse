import sparse
import  numpy as np

def sliding_window_process(inputdata, time_step):
    # datainput shape --> (data_len, 30)
    data_len = len(inputdata)
    sliding_len = data_len-(time_step-1)
    out_arr = np.zeros([sliding_len, time_step, 45])

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

def sparse_martix_to_xxyyzz(voxel_data):
    out_data = np.zeros([len(voxel_data),45])
    for i in range(len(voxel_data)):
        tmp_out = np.zeros([1,45])
        tmp = sparse.COO(voxel_data[i])
        tmp_len = len(tmp.coords[2])
        tmp_out[0,0:tmp_len] = tmp.coords[1]
        tmp_out[0,tmp_len:tmp_len*2] = tmp.coords[2]
        tmp_out[0,tmp_len*2: tmp_len*3] = tmp.coords[3]
        out_data[i] = tmp_out[0]
    return  out_data

def sparse_martix_to_xyzxyz(voxel_data):
    out_data = np.zeros([len(voxel_data),45])
    for i in range(len(voxel_data)):
        tmp_out = np.zeros([1,45])
        tmp = sparse.COO(voxel_data[i])
        for index in range(len(tmp.coords[2])):
            tmp_out[0,index*3] = tmp.coords[1][index]
            tmp_out[0,index*3+1] = tmp.coords[2][index]
            tmp_out[0,index*3+2] = tmp.coords[3][index]
        out_data[i] = tmp_out[0]
    return  out_data


if __name__ == '__main__' :

    # Gesture = ["circle"]
    Gesture = ["circle", "eight", "rectangle", "up", "down", "left", "right"]
    PATH = "D:/thumouse_training_data/"
    dir_name = 'transfer_0'
    head_path = PATH + dir_name+'/'
    save_path = head_path + "sliding_data/"
    name = '_scr_'+ dir_name
    time = 0
    for i in Gesture:
        for j in range(2, 4):

            path = head_path + i + "/time" + str(j) + "/"
            out_cam_p = np.load(path + 'out_cam_p' + name + '.npy', allow_pickle=True)
            radar = np.load(path + 'out_radar_p' + name + '.npy', allow_pickle=True)
            # out_radar_p = np.load(path + 'out_center_p' + name + '.npy', allow_pickle=True) 　#點雲重心資料

            # radar = radar[24:984] # for moving average mid out
            radar = radar[20:980] # for moving average mid out
            out_cam_p = out_cam_p[20:980]

            sparse_out_xyzxyz = sparse_martix_to_xyzxyz(radar)
            sparse_out_xxyyzz = sparse_martix_to_xxyyzz(radar)
            outdata_xyzxyz = sliding_window_process(sparse_out_xyzxyz, time_step=3)
            outdata_xxyyzz = sliding_window_process(sparse_out_xxyyzz, time_step=3)
            outcamp = sliding_window_process_camera(out_cam_p[:,:,8], time_step=3)

            # np.save(save_path+ i +"_time" + str(j) + "_radar_xyzxyz.npy", outdata_xyzxyz)
            # np.save(save_path+ i +"_time" + str(j) + "_radar_xxyyzz.npy", outdata_xxyyzz)
            # np.save(save_path+ i +"_time" + str(j) + "_cam.npy", outcamp)
#


