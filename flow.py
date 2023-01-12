import os
import sys
import pickle
import datetime
import warnings

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold

import tensorflow as tf
import madmom

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

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
                )

preprocessor = cnn_preprocessor()

def cnn_normalize(frames):
    inv_std = np.load("models/bock2013pret_inv_std.npy")
    mean = np.load("models/bock2013pret_mean.npy")
    frames_normalized = (frames - np.reshape(mean, (1,80,3)))*np.reshape(inv_std, (1,80,3))
    return frames_normalized

def main(finetune, extend, dropout_p, relu, learning_r=0.001):

    ds0 = Dataset("initslurtest")
    ds1 = Dataset("slurtest_add_1")
    ds2 = Dataset("slurtest_add_2")
    ds3 = Dataset("slurtest_test")

    audio_fnames = ds0.get_audio_paths() + ds1.get_audio_paths() + ds2.get_audio_paths() + ds3.get_audio_paths()
    label_fnames = ds0.get_annotation_paths() + ds1.get_annotation_paths() + ds2.get_annotation_paths() + ds3.get_annotation_paths()

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
        mode=None,
        standard=False,
        mean=None,
        std=None
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
                frames = mm_frames_normalized[file_i]
            elif mode=="use_raw_frames":
                # No normalization
                frames = mm_proc_frames[file_i]    
            else:
                # Compute frames
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fxn()
                    frames = cnn_normalize(preprocessor(fname))

            if standard:
                frames = (frames-mean)/std
                
            #print("Frame size: ", frames.shape[0])

            # Retrieve onsets 
            onsets = onset_vectors[file_i]
            #print("Computed frames of size ", frames.shape)
            #print("Onset vectors have len ", len(onsets))

            # Sample a set of indices (defined from audio start,
            # that is CONTEXT values counted from x array start)
            if sampling:
                focus_idx = np.random.choice(
                    np.arange(frames.shape[0]-2*CONTEXT-1), 
                    size=batch_size
                )
                #print("Sampled focus idx between ", 0, " and ", frames.shape[0]-2*CONTEXT-1)
            else:
                #print("Focus idx from ", frame_p, " to ", frame_p+batch_size)
                focus_idx = np.arange(frame_p, frame_p+batch_size)
            

            # Segmentation
            x = [frames[focus:focus+2*CONTEXT+1,:,:] for focus in focus_idx]
            x = np.transpose(np.stack(x, 0), [0,2,1,3])
            #print("Segmented x has shape ", x.shape)
            if x.shape[0] != batch_size:
                print("Delivering less than batch-size")

            # Labels
            y = onsets[focus_idx]
            yield (x, y)

            if not sampling:
                if frame_p + 2*batch_size >= frames.shape[0]-2*CONTEXT-1:
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
    #random_seed = 119
    #n_splits =  5
    #kf = KFold(n_splits=n_splits, shuffle=True, random_state=True)
    #folds = list(kf.split(np.arange(len(audio_fnames))))

    # Partitioned by musician:
    sa_recs = list(np.arange(19)) + [23, 25, 28, 32, 36, 37, 45, 46]
    fk_recs = [22, 29, 30, 33, 44, 47]
    ir_recs = np.setdiff1d(list(np.arange(49)), sa_recs + fk_recs)

    random_seed = 119
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    folds = list(skf.split(
        np.concatenate((sa_recs, ir_recs)), # Indices in devset
        np.concatenate((np.zeros(len(sa_recs)), np.ones(len(ir_recs)))) # Boolean whether recs are played by a certain musician
    ))

    
    # Custom splits
    n_splits = 1
    folds = [[np.arange(len(audios)), []]]
    #sa_choice = np.random.choice(sa_recs, size=len(sa_recs), replace=False)
    #ir_choice = np.random.choice(ir_recs, size=len(ir_recs), replace=False)
    #bp_sa = int(len(sa_recs)*0.9)
    #bp_ir = int(len(ir_recs)*0.9)
    #folds = [[list(sa_choice[:bp_sa]) + list(ir_recs[:bp_ir]), list(sa_choice[bp_sa:]) + list(ir_recs[bp_ir:])]]

    # Precompute statistics:
    means_per_fold = []
    std_per_fold = []
    for train_idx, test_idx in folds:
        train_frames = np.concatenate([mm_proc_frames[i] for i in train_idx]) 
        mean_train = train_frames.mean(0)
        std_train = train_frames.std(0, ddof=1)
        means_per_fold.append(np.expand_dims(mean_train, axis=0))
        std_per_fold.append(np.expand_dims(std_train, axis=0))
    # Positive class_weight
    W = np.sum([len(vec) for vec in onset_vectors])/np.sum([vec.sum() for vec in onset_vectors])

    def compute_steps(idx, bs):
        song_sizes = np.array([len(f) for f in mm_frames_normalized])[idx]-2*CONTEXT-1
        steps_per_song = np.floor_divide(song_sizes, bs)
        return np.sum(steps_per_song)

    def wbce(y_true, y_pred):
        y_pred = tf.keras.backend.clip(y_pred, 1e-7, 1-1e-7)
        logits = tf.keras.backend.log(y_pred/(1-y_pred))
        return tf.nn.weighted_cross_entropy_with_logits(
            y_true, logits, W
        )

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam()
    metrics = [wbce]

    datasets = "full"
    continue_run = False
    training_mode = "all" # REMEMBER TO CHANGE
    check_at_epoch = 20 # REMEMBER TO CHANGE

    save = True # REMEMBER TO CHANGE
    # REMEMBER TO CHANGE
    save_path = "results/cnn-training-230111/" # TODO - automatically
    n_epochs = 100 # REMEMBER TO CHANGE
    #learning_r = 0.001 as function parameter
    bs = 512
    steps_per_epoch = 0 # is set later
    val_steps_per_epoch = 100 # needed?
    nogen = False
    sampling = False
    cw_dict = {0: 1., 1: W}

    standard = True # keep in mind on which data format statistics are computed 
    mode = 'use_prep_frames' # Preparing by BN layer/"CNN normalization"
    #mode = 'use_raw_frames' # No preparing


    training_name = "{}-{}-{}eps-{}-{}-{}-dropout{:.1f}{}".format(
        datasets, 
        str(learning_r),
        #"nogen" if nogen else ("sample" if sampling else "seq"),
        n_epochs,
        "standard" if standard else "nostandard",
        "finetune" if finetune else "trainable",
        "extend" if extend else "noextend",
        dropout_p,
        "-relu" if relu else ""
    )

    if isinstance(training_mode, int):
        fold = training_mode
    else:
        fold = 0

    while fold < n_splits:
        print()
        print("Fold {}/{} ---------".format(fold, n_splits))
        train_idx = folds[fold][0]
        test_idx = folds[fold][1]
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
        if standard:
            mean = means_per_fold[fold]
            std = std_per_fold[fold]
        else:
            mean, std = None, None

        # Model
        if not continue_run:
            tf.keras.backend.clear_session()
        (model, norm_layer)=get_model(finetune=finetune, extend=extend, dropout_p=dropout_p, relu=relu)
        model.compile(optimizer=optimizer,
                    loss=loss_fn,
                    metrics=metrics
        )
                    
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
                mode=mode,
                standard=standard, mean=mean, std=std,
            )
            y = None
            validation_data = data_generator(
                batch_size=bs, 
                steps_per_epoch=val_steps_per_epoch, 
                epochs=n_epochs,
                idx=test_idx,
                sampling=sampling,
                mode=mode,
                standard=standard, mean=mean, std=std,
            )

        checkpoint_path = save_path + 'fold_{}_{}'.format(fold,training_name)+"_cp_{epoch:04d}.ckpt"
        if check_at_epoch is None:
            cp_callback=[]
        else:
            cp_callback = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
                                                        save_weights_only=True,
                                                        save_freq=int(steps_per_epoch*check_at_epoch))]
        # Training
        history = model.fit(
            x = x, y = y, 
            steps_per_epoch = steps_per_epoch,
            epochs          = n_epochs,
            # Validation data
            validation_data = None, # validation_data, # REMEMBER TO CHANGE
            #validation_steps  = val_steps_per_epoch,
            class_weight = cw_dict,
            callbacks=cp_callback,
            verbose=1
        )

        # Saving
        if save:
            model.save(save_path + 'fold_{}_{}_model'.format(fold, training_name))
            with open(save_path + 'fold_{}_{}_history.pickle'.format(fold, training_name), 'wb') as file_pi:
                pickle.dump(history.history, file_pi)
        
        
        if training_mode != "all":
            break
        fold += 1

if __name__=="__main__":
    
    main(finetune=False, extend=False, dropout_p=0.8, relu=False, learning_r=0.001)
    main(finetune=True, extend=False, dropout_p=0.8, relu=False, learning_r=0.001)
    """
    for relu in [True, False]:
        for dropout_p in [0,0.3,0.5]:
            for mode in ["normal", "finetune", "extend"]:
                if mode=="normal":
                    main(finetune=False, extend=False, dropout_p=dropout_p, relu=relu)
                elif mode=="finetune":
                    main(finetune=True, extend=False, dropout_p=dropout_p, relu=relu)
                elif mode=="extend":
                    main(finetune=True, extend=True, dropout_p=dropout_p, relu=relu)
    """
