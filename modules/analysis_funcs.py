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

def merging_module(note_onsets, hard_onsets, tol_sec):
    matching = []
    lacking = []
    for t in range(len(note_onsets)):
        matches = np.where(np.abs(hard_onsets-note_onsets[t])<tol_sec, 1, 0)
        if np.sum(matches)>0:
            matching.append(t)
        else:
            lacking.append(t)
            # Returns indices!
    return np.array(matching), np.array(lacking)

def slur_alignment(hard_idx, soft_idx):
    h = np.zeros((len(hard_idx)+len(soft_idx)))
    slur_onehot = np.zeros((len(hard_idx)+len(soft_idx), 4))
    h[hard_idx] = 1

    h_shift = h[1:]
    h_orig = h[:-1]
    sos = np.where(np.logical_and(h_orig==1, h_shift==0))[0]
    inside = np.where(np.logical_and(h_orig==0, h_shift==0))[0]
    eos = np.where(np.logical_and(h_orig==0, h_shift==1))[0]
    det = np.where(np.logical_and(h_orig==1, h_shift==1))[0]
    slur_onehot[sos, 0] = 1
    slur_onehot[inside, 1] = 1
    slur_onehot[eos, 2] = 1
    slur_onehot[det, 3] = 1
    return slur_onehot

def aubio_peakpicker_do(hfc, threshold=0.2, win_pre=1, win_post=5):
    
    out = np.zeros_like(hfc)
    out_thres = np.zeros_like(hfc)
    onset_keep = np.zeros(7)
    onset_proc = np.zeros(7)
    onset_peek = np.zeros(3)
    thresholded = np.zeros(1)
    mean = 0.
    median = 0.
    j = 0
    
    for n in range(len(hfc)-win_post-1):
        # /* push new novelty to the end */
        # fvec_push(onset_keep, onset->data[0]);
        onset_proc = np.concatenate([onset_proc, [hfc[n]]])[1:]
        #/* store a copy */
        #fvec_copy(onset_keep, onset_proc);

        #/* filter this copy */
        #aubio_filter_do_filtfilt (p->biquad, onset_proc, scratch);
        sos_coeffs = np.array([0.15998789, 0.31997577, 0.15998789, 1, -0.59488894, 0.23484048])
        #onset_proc = sosfilt(sos_coeffs, onset_proc)

        #/* calculate mean and median for onset_proc */
        mean = np.mean(onset_proc)

        #/* copy to scratch and compute its median */
        # fvec_copy(onset_proc, scratch);
        median = np.median(onset_proc)

        
        #/* shift peek array */
        # for (j = 0; j < 3 - 1; j++)
        for j in range(0,3-1):     
            #onset_peek->data[j] = onset_peek->data[j + 1];
            onset_peek[j] = onset_peek[j+1]
        #/* calculate new tresholded value */
        thresholded = onset_proc[win_post] - median - mean * threshold;
        onset_peek[2] = thresholded;
        if onset_peek[1]>0 and onset_peek[0]<onset_peek[1] and onset_peek[1]>onset_peek[2]:
            out[n] = 1
        else:
            out[n] = 0
        out_thres[n] = thresholded
         
    return out, out_thres

def aubio_postprocessing(onsets_oh, sig, final_shift=5, db_thres=-80, min_ioi_frames=3):
    frames = madmom.audio.signal.FramedSignal(sig, frame_size=512, hop_size=256)
    spl = madmom.audio.signal.sound_pressure_level(frames)
    ons = np.where(onsets_oh==1)[0]
    silence_gate = np.where(spl[ons]>db_thres)[0]
    ons =ons[silence_gate]
    valid_idx = np.where(np.ediff1d(ons)>=min_ioi_frames)[0]
    ons = ons[valid_idx]-final_shift
    return ons