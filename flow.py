import os
import pickle
import datetime
import warnings

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split

import tensorflow as tf
import madmom

from segment import segment_spectrogram
from modules.labels import get_label_vector
from modules.madmom_cnn_prep import cnn_preprocessor
from datasets import Dataset
from analyze_detection import evaluate, f_score
from models.bock2013pret import get_model

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

SR = 44100
FPS = 100
CONTEXT = 7

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

preprocessor = cnn_preprocessor()

def cnn_normalize(frames):
    inv_std = np.load("models/bock2013pret_inv_std.npy")
    mean = np.load("models/bock2013pret_mean.npy")
    frames_normalized = (frames - np.reshape(mean, (1,80,3)))*np.reshape(inv_std, (1,80,3))
    return frames_normalized

ds0 = Dataset("initslurtest")
ds1 = Dataset("slurtest_add_1")

audio_fnames = ds0.get_audio_paths() + ds1.get_audio_paths()
label_fnames = ds0.get_annotation_paths() + ds1.get_annotation_paths()

audios = [madmom.audio.signal.load_wave_file(filename)[0] for filename in audio_fnames]
sample_rates = [madmom.audio.signal.load_wave_file(filename)[1] for filename in audio_fnames]
onset_schedules = [np.loadtxt(label_fname, usecols=0) for label_fname in label_fnames]
onset_vectors = [get_label_vector(sched, len(audio)/sr, FPS)
    for (sched, audio, sr) in zip(onset_schedules, audios, sample_rates)
]

mm_proc_frames = [preprocessor(fname) for fname in audio_fnames]
mm_frames_normalized = [cnn_normalize(frame_set) for frame_set in mm_proc_frames]

def data_generator(
    batch_size,
    steps_per_epoch,
    epochs,
    idx, 
    sampling=True,
    mode=None
):
    
    #for _ in range(steps_per_epoch * epochs):
    if not sampling:
        ep = 0
        file_p = 0
        frame_p = 0
    
    while True:
        # Select indices for training or test
        if sampling:
            file_i = np.random.choice(idx)
        else:
            file_i = idx[file_p]
        
        #print("Selected file index: ", file_i)
        fname = audio_fnames[file_i]
        
        if mode=="use_prep_frames":
            frames_normalized = mm_frames_normalized[file_i]
            
        else:
            # Compute frames
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fxn()
                frames_normalized = cnn_normalize(preprocessor(fname))
        #print("Frame size: ", frames_normalized.shape[0])

        # Retrieve onsets 
        onsets = onset_vectors[file_i]
        #print("Computed frames of size ", frames_normalized.shape)
        #print("Onset vectors have len ", len(onsets))

        # Sample a set of indices (defined from audio start,
        # that is CONTEXT values counted from x array start)
        if sampling:
            focus_idx = np.random.choice(
                np.arange(frames_normalized.shape[0]-2*CONTEXT-1), 
                size=batch_size
            )
            #print("Sampled focus idx between ", 0, " and ", frames_normalized.shape[0]-2*CONTEXT-1)
        else:
            #print("Focus idx from ", frame_p, " to ", frame_p+batch_size)
            focus_idx = np.arange(frame_p, frame_p+batch_size)
        

        # Segmentation
        x = [frames_normalized[focus:focus+2*CONTEXT+1,:,:] for focus in focus_idx]
        x = np.transpose(np.stack(x, 0), [0,2,1,3])
        #print("Segmented x has shape ", x.shape)
        if x.shape[0] != batch_size:
            print("Delivering less than batch-size")

        # Labels
        y = onsets[focus_idx]
        yield (x, y)

        if not sampling:
            if frame_p + 2*batch_size >= frames_normalized.shape[0]-2*CONTEXT-1:
                if file_p == len(idx) - 1:
                    ep += 1
                    print("Generator reached end of epoch. Resetting...")
                    file_p = 0
                    frame_p = 0
                else:
                    file_p += 1
                    frame_p = 0
            else:
                frame_p += batch_size

# K-Fold:
random_seed = 119
n_splits =  5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=True)
kf_gen = list(kf.split(np.arange(len(audio_fnames))))

def compute_steps(idx, bs):
    song_sizes = np.array([len(f) for f in mm_frames_normalized])[idx]-2*CONTEXT-1
    steps_per_song = np.floor_divide(song_sizes, bs)
    return np.sum(steps_per_song)


loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam()
metrics = [
    tf.keras.metrics.TruePositives(name='tp', thresholds=0.5),
    tf.keras.metrics.TrueNegatives(name='tn', thresholds=0.5),
    tf.keras.metrics.FalsePositives(name='fp', thresholds=0.5),
    tf.keras.metrics.FalseNegatives(name='fn', thresholds=0.5),
]

continue_run = False
training_mode = "all" # REMEMBER TO CHANGE
standard = False
save = True # REMEMBER TO CHANGE
training_name = "added-sample-gen-nostandard" # REMEMBER TO CHANGE
date_today = "220409" # TODO - automatically
n_epochs = 50 # REMEMBER TO CHANGE
bs = 256
steps_per_epoch = 400
val_steps_per_epoch = 30
nogen = False 
sampling = True

if standard:
    with open('results/cnn-training-220331/mean_by_fold.pickle', 'rb') as file_pi:
        means = pickle.load(file_pi)
    with open('results/cnn-training-220331/std_by_fold.pickle', 'rb') as file_pi:
        stds = pickle.load(file_pi)

if isinstance(training_mode, int):
    fold = training_mode
else:
    fold = 0

while fold < n_splits:
    print()
    print("Fold {}/{} ---------".format(fold, n_splits))
    train_idx = kf_gen[fold][0]
    test_idx = kf_gen[fold][1]
    print("Train indices: ", train_idx)
    print("Test indices: ", test_idx)

    # Data
    if nogen:
        X_train, X_test = [
            np.concatenate([X[i] for i in idx]) 
            for idx in (train_idx, test_idx)
        ]
        y_train, y_test = [
            np.concatenate([onset_vectors[i] for i in idx]) 
            for idx in (train_idx, test_idx)
        ]

    #train_onset_ratio = y_train.sum()/len(y_train)

    # Normalize with training set statistics
    #if standard:
    #    X_train = (X_train - means[fold])/stds[fold]
    #    X_test = (X_test - means[fold])/stds[fold]

    # Model
    if not continue_run:
        tf.keras.backend.clear_session()
    (model, norm_layer)=get_model(finetune=False)
    
    model.compile(optimizer=optimizer,
                loss=loss_fn,
                metrics=metrics)
                
    if not sampling:
        steps_per_epoch = compute_steps(train_idx, bs)
        val_steps_per_epoch = compute_steps(test_idx, bs)

    if nogen:
        x = X_train
        y = y_train
        steps_per_epoch = None
        validation_data = (X_test, y_test)
    else:
        x = data_generator(
            batch_size=bs, 
            steps_per_epoch=steps_per_epoch, 
            epochs=n_epochs,
            idx=train_idx,
            sampling=sampling,
            mode='use_prep_frames'
        )
        y = None
        validation_data = data_generator(
            batch_size=bs, 
            steps_per_epoch=val_steps_per_epoch, 
            epochs=n_epochs,
            idx=test_idx,
            sampling=sampling,
            mode='use_prep_frames'
        )


    # Training
    history = model.fit(
        x = x, y = y, 
        steps_per_epoch = steps_per_epoch,
        epochs          = n_epochs,
        # Validation data
        validation_data = validation_data,
        validation_steps  = val_steps_per_epoch,
        class_weight = {0: 1., 1: 1/0.035},
        verbose=1
    )

    # Saving
    if save:
        model.save('results/cnn-training-{}/fold_{}_{}_model'.format(date_today, fold, training_name))
        with open('results/cnn-training-{}/fold_{}_{}_history.pickle'.format(date_today, fold, training_name), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    
    
    if training_mode != "all":
        break
    fold += 1

