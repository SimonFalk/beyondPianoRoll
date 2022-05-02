import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import madmom
import mir_eval

from modules.labels import get_label_vector
from modules.madmom_cnn_prep import cnn_preprocessor
from datasets import Dataset
from modules.analysis_funcs import get_idx_to_fold, get_segmented_data, aubio_peakpicker_do, aubio_postprocessing
from analyze_detection import evaluate
from modules.energy_based import legato_mg

FPS = 100
CONTEXT = 7

# Load Madmom normalization
def cnn_normalize(frames):
    inv_std = np.load("models/bock2013pret_inv_std.npy")
    mean = np.load("models/bock2013pret_mean.npy")
    frames_normalized = (frames - np.reshape(mean, (1,80,3)))*np.reshape(inv_std, (1,80,3))
    return frames_normalized

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

ds0 = Dataset("initslurtest")
ds1 = Dataset("slurtest_add_1")

audio_fnames = ds0.get_audio_paths() + ds1.get_audio_paths()
label_fnames = ds0.get_annotation_paths() + ds1.get_annotation_paths()

audios = [madmom.audio.signal.load_wave_file(filename)[0] for filename in audio_fnames]
sample_rates = [madmom.audio.signal.load_wave_file(filename)[1] for filename in audio_fnames]
onset_schedules = [np.loadtxt(label_fname, usecols=0) for label_fname in label_fnames]

base_path = "results/cnn-training-220426/"
model_name = "ab-seq-90eps-nostandard-trainable-noextend-dropout0.3"
model = tf.keras.models.load_model(base_path + 'fold_{}_{}_model'.format(0, model_name))
TOL = 0.025

def evaluate_set(idx, TOL=0.025, f_rate=0.01, **kwargs):
    fs = []
    rs = []
    ps = []
    CD_list = []
    FN_list = []
    FP_list = []
    for r in idx:
        rec_name = os.path.basename(audio_fnames[r])
        x = get_segmented_data(audio_fnames[r])
        out = model.predict(x)
        peaks = madmom.features.onsets.peak_picking(
                                        activations=out, 
                                        **kwargs
        )[0].astype(np.float32)*f_rate
        [CD,FN,FP,doubles,merged] = evaluate(onset_schedules[r], peaks, tol_sec=TOL)
        CD_list.append(CD)
        FN_list.append(FN)
        FP_list.append(FP)
        scores = mir_eval.onset.evaluate(onset_schedules[r], peaks, window=TOL)
        fs.append(scores["F-measure"])
        ps.append(scores["Precision"])
        rs.append(scores["Recall"])
    f_tot = np.sum(CD_list)/(np.sum(CD_list)+.5*(np.sum(FP_list) + np.sum(FN_list)))
    p_tot = np.sum(CD_list)/(np.sum(CD_list)+np.sum(FP_list))
    r_tot = np.sum(CD_list)/(np.sum(CD_list)+np.sum(FN_list))
    return [np.mean(fs), np.mean(ps), np.mean(rs), np.std(fs), np.std(ps), np.std(rs), f_tot, p_tot, r_tot]


thress = np.arange(0.3,0.75,0.1)
smooths = [0,5,7]
pres = [0,1,3]
posts = [0,3,5]


results = np.zeros((len(thress), len(smooths), len(pres), len(posts), 9))
for i, threshold in enumerate(thress):
    for j, smooth in enumerate(smooths):
        for k, n_pre in enumerate(pres):
            for l, n_post in enumerate(posts):
                metrics = np.array(evaluate_set(np.arange(len(audio_fnames)),
                                        threshold=threshold, 
                                        smooth=smooth, 
                                        pre_avg=n_pre, 
                                        post_avg=n_post, 
                                        pre_max=n_pre, 
                                        post_max=n_post))
                results[i,j,k,l] = metrics
                print(metrics)
np.save(file="results/computed/pp_metrics.npy", arr=results)
