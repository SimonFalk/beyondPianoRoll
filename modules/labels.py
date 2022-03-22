import numpy as np

def get_label_vector(onset_times, length_s, HOP, sr, fuzzy=False):
    times = np.arange(0, length_s, HOP/sr)
    a = np.reshape(onset_times, (-1,1))
    b = np.reshape(times, (1,-1))
    onset_onehot = np.sum(np.abs(a - b) < HOP/(2*sr), 0)
    onset_wide = np.sum(np.abs(a - b) < 3*HOP/(2*sr), 0)
    onset_fuzzy = 0.25*onset_wide + 0.25*onset_onehot
    
    if fuzzy: 
        return onset_fuzzy
    else:
        return onset_onehot