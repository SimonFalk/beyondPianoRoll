import numpy as np
import madmom
from .madmom_cnn_prep import cnn_normalize, cnn_preprocessor


def get_idx_to_fold(folds):
    itf = {}
    for (fold_i, fold) in enumerate(folds):
        for index in fold[1]:
            # Loop through test set
            itf[index] = fold_i
    
    return itf

def get_segmented_data(path, do_cnn_normalize=True, CONTEXT=7):
    preprocessor = cnn_preprocessor()
    frames = preprocessor(path)
    if do_cnn_normalize:
        frames = cnn_normalize(frames)
    x = [
        frames[i-CONTEXT:i+CONTEXT+1,:,:] 
        for i in range(CONTEXT, frames.shape[0]-CONTEXT)
    ]
    x = np.stack(x, 0)
    x = np.transpose(x, [0,2,1,3])
    return x

def get_test_peaks(activations, f_rate, threshold=0.5, kernel_size=5, n_pre=1, n_post=5):
    return madmom.features.onsets.peak_picking(
                                        activations=activations, 
                                        threshold=threshold, 
                                        smooth=kernel_size, 
                                        pre_avg=n_pre, 
                                        post_avg=n_post, 
                                        pre_max=n_pre, 
                                        post_max=n_post
    )[0].astype(np.float32)*f_rate

