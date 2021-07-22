import  numpy as np
x = np.zeros([40])
for i in range(40):
   x[i] = i

data_len = len(x)
sliding_len = data_len - 2
for i in range(sliding_len):
    tmp_pre = i
    tmp = i+1
    tmp_next = i+2
    print(tmp_pre,tmp,tmp_next)

data_len = len(x)
sliding_len = data_len - 5
for i in range(sliding_len):
    tmp_pre = i
    tmp = i+1
    tmp_next = i+2
    tmp_next1 = i+3
    tmp_next2 = i+4
    tmp_next3 = i+5
    print(tmp_pre,tmp,tmp_next,tmp_next1,tmp_next2,tmp_next3)
