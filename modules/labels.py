import numpy as np

def invert_permutation(p):
    """Return an array s with which np.array_equal(arr[p][s], arr) is True.
    The array_like argument p must be some permutation of 0, 1, ..., len(p)-1.
    """
    p = np.asanyarray(p) # in case p is a tuple, etc.
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s

s = invert_permutation(perm)

def get_label_table(Xlist,marg):
    label_table = np.array([], dtype=np.int64).reshape(2*marg+1,0)
    cur = 0
    for i in range(len(Xlist)):
        label_table = np.concatenate((
            label_table,
            np.stack([np.arange(cur+k, cur+Xlist[i].shape[0]-marg*2+k) for k in range(2*marg+1)])
            
        ), 1)
        cur+=Xlist[i].shape[0]
    n_tot = label_table.shape[1]
    return label_table

def shuffle_frames(Xlist,ylist, label_table, marg=7, shuffle=True):
    Xconc = np.concatenate(Xlist)
    yconc = np.concatenate(ylist)
    if shuffle:
        perm = np.random.permutation(len(yconc))
    else:
        perm = np.arange(len(yconc))
    print(perm)
    Xp =np.concatenate(Xlist)[label_table[marg,:].reshape(-1).astype(int)][perm]
    #Xp = np.concatenate(X)[label_table[marg,:].reshape(-1).astype(int)][perm]
    yp = np.concatenate(ylist)[perm]
    lt = label_table[:,perm]
    return Xp, yp, perm
    


def get_label_vector(onset_times, length_s, fps, fuzzy=False):
    times = np.arange(0, length_s, 1/fps)
    a = np.reshape(onset_times, (-1,1))
    b = np.reshape(times, (1,-1))
    onset_onehot = np.sum(np.abs(a - b) < 0.5/fps, 0)
    #onset_wide = np.sum(np.abs(a - b) < 3*HOP/(2*sr), 0)
    #onset_fuzzy = 0.25*onset_wide + 0.25*onset_onehot
    
    if fuzzy: 
        return onset_fuzzy
    else:
        return onset_onehot
