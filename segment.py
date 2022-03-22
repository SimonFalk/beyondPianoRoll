import os
import sys
import pickle
import numpy as np



def segment_spectrogram(spect, n_margin):
    data = []
    n_samples = spect.shape[1]
    for i in range(n_margin, n_samples - n_margin - 1):
        data.append(spect[:,i-n_margin:i+n_margin+1])

    data = np.stack(data)
    return data

if __name__=="__main__":

    if len(sys.argv) == 1:
        in_dir = '../DoReMir/initslurtest_vn/spectrogram/'
        out_dir = '../DoReMir/initslurtest_vn/frames/'
    else:
        in_dir = sys.argv[1]
        out_dir = sys.argv[2]

    N_MARGIN = 7

    for file in os.listdir(in_dir):
        if not file.endswith(".pickle"):
            continue
        
        fname = os.path.join(in_dir, file)
        print(fname)
        with open(fname, 'rb') as handle:
            spects = pickle.load(handle)
        data = []
        for spect in spects:
            segmented = segment_spectrogram(spect, N_MARGIN)
            data.append(segmented)
        data = np.stack(data, axis=-1)
        print(data.shape)
        out_name = os.path.join(out_dir, file)
        
        with open(out_name, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        