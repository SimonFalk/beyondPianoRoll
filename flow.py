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

def data_generator(
    batch_size,
    steps_per_epoch,
    epochs,
    idx,
    mode = 0 # 0 for training, 1 for test
):
    
    for _ in range(steps_per_epoch * epochs):

        # Select indices for training or test
        file_i = np.random.choice(idx)
        #print("Selected file index: ", file_i)
        fname = audio_fnames[file_i]
        
        # Compute frames
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fxn()
            frames_normalized = cnn_normalize(preprocessor(fname))

        # Retrieve onsets 
        onsets = onset_vectors[file_i]
        #print("Computed frames of size ", frames_normalized.shape)
        #print("Onset vectors have len ", len(onsets))

        # Sample a set of indices (defined from audio start,
        # that is CONTEXT values counted from x array start)
        focus_idx = np.random.choice(
            np.arange(frames_normalized.shape[0]-2*CONTEXT-1), 
            size=batch_size
        )
        #print("Sampled focus idx between ", 0, " and ", frames_normalized.shape[0]-2*CONTEXT-1)

        # Segmentation
        x = [frames_normalized[focus:focus+2*CONTEXT+1,:,:] for focus in focus_idx]
        x = np.transpose(np.stack(x, 0), [0,2,1,3])
        #print("Segmented x has shape ", x.shape)

        # Labels
        
        y = onsets[focus_idx]
        yield (x, y)

# K-Fold:
random_seed = 119
n_splits =  5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=True)
kf_gen = list(kf.split(np.arange(len(audio_fnames))))


loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam()
metrics = [
    tf.keras.metrics.TruePositives(name='tp', thresholds=0.5),
    tf.keras.metrics.TrueNegatives(name='tn', thresholds=0.5),
    tf.keras.metrics.FalsePositives(name='fp', thresholds=0.5),
    tf.keras.metrics.FalseNegatives(name='fn', thresholds=0.5),
]

training_mode = 0
standard = False
save = True
training_name = "added-sample-nostandard"
date_today = "220407"
n_epochs = 50
steps_per_epoch = 100
val_steps_per_epoch = 30
bs = 256

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

    # Normalize with training set statistics

    # Model
    tf.keras.backend.clear_session()
    (model, norm_layer)=get_model(finetune=False)
    model.compile(optimizer=optimizer,
                loss=loss_fn,
                metrics=metrics)

    # Training
    history = model.fit(
        x = data_generator(
            batch_size=bs, 
            steps_per_epoch=steps_per_epoch, 
            epochs=n_epochs,
            idx=train_idx
        ),
        batch_size      = bs,
        steps_per_epoch = steps_per_epoch,
        epochs          = n_epochs,
        
        # Validation data
        validation_data = tf.data.Dataset.from_generator(
            lambda: data_generator(
                batch_size=bs, 
                steps_per_epoch=val_steps_per_epoch, 
                epochs=1,
                idx=test_idx
            ),
            output_signature = (
                tf.TensorSpec(shape = (bs, 80 , 15, 3), dtype = tf.float64),
                tf.TensorSpec(shape = (bs),     dtype = tf.float64),
            )
        ),
        validation_batch_size = bs,
        validation_steps      = val_steps_per_epoch,
        class_weight = {0: 1., 1: 1/0.035},
        verbose=1
    )

    # Saving
    if save:
        model.save('results/cnn-training-{}/{}_fold_{}_model'.format(date_today, fold, training_name))
        with open('results/cnn-training-{}/{}_fold_{}_history.pickle'.format(date_today, fold, training_name), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    
    
    if training_mode != "all":
        break
    fold += 1

