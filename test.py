import numpy as np

_s_l_stop_mask = np.array([[0, 1,0,0], [0,0,0,0], [1,0,0,0], [1,1,0,0],[0,0,1,0]])




def first_true1(a, default):
    di=np.full(a.shape[0], -1)
    for i in range(len(a)):
        idx = np.where(a[i] > 0)
        try:
            di[i] = idx[0][0]
        except IndexError:
            di[i] = default

    return di

def first_true2(a, default):
    di = np.full(a.shape[0], -1)
    for i in range(len(a)):
        idx=np.argmax(a[i])
        if idx>0:
            di[i]=idx
        else:
            di[i]=default

    return di

def first_true4(a, default):
    di = np.full(a.shape[0], -1)
    for i, ele in enumerate(np.argmax(a,axis=1)):
        if ele==0 and a[i][0]==0:
            di[i]=default
        else:
            di[i]=ele

    return di

_s_l_stop = np.any(_s_l_stop_mask, axis=1)
print(_s_l_stop)
_s_l_ext_idx = first_true1(_s_l_stop_mask,3)
print(_s_l_ext_idx)
_s_l_ext_idx = first_true2(_s_l_stop_mask,3)
print(_s_l_ext_idx)
_s_l_ext_idx = first_true4(_s_l_stop_mask,3)
print(_s_l_ext_idx)


# _s_l_ext_idx = np.full((4), 3)
# _s_l_stop = np.any(_s_l_stop_mask, axis=1)
# _s_l_stop_idxs, _b = np.nonzero(_s_l_stop_mask)
# _s_l_ext_idx[_s_l_stop_idxs] = _b
# print(_s_l_stop)
# print(_s_l_ext_idx)
