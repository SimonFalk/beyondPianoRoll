import os
import sys
import numpy as np
import pickle
import librosa
from librosa.feature import melspectrogram

if len(sys.argv) == 1:
    in_dir = '../DoReMir/initslurtest_vn/initslurtest_vn_wav/'
    out_dir = '../DoReMir/initslurtest_vn/spectrogram/'
else:
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]

HOP = 440
MELS = 80
SR = 44100

def multi_window_spectrogram(audio_tensor, win_sizes, hop_size, mels, fmin, fmax, sr=SR):
    spects = []
    for win_size in win_sizes:
        spectrogram = melspectrogram(y=signal, 
            sr=sr,
            hop_length=hop_size,
            win_length=win_size,
            n_fft=win_size,
            n_mels=mels,
            fmin=fmin,
            fmax=fmax
        )
        spects.append(np.log(spectrogram + np.finfo(float).eps))
    return spects

for file in os.listdir(in_dir):
    if not file.endswith(".wav"):
        continue
    
    fname = os.path.join(in_dir, file)
    print(fname)
    signal, sample_rate = librosa.load(fname, sr=SR)
    spects = multi_window_spectrogram(signal, [1024, 2048, 4096], HOP, MELS, 27.5, 16000.0)
    print(spects[0].shape)
    out_name = os.path.join(out_dir, file[:-4] + '.pickle')
    with open(out_name, 'wb') as handle:
        pickle.dump(spects, handle, protocol=pickle.HIGHEST_PROTOCOL)