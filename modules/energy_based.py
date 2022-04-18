import numpy as np
import madmom
from madmom.features.onsets import CNNOnsetProcessor, OnsetPeakPickingProcessor

def bock_onsets(file, ada_comb=True):
    typ_dist = None
    cnn = CNNOnsetProcessor()
    act_fn = cnn(file)
    pp = OnsetPeakPickingProcessor()
    onsets = pp.process_offline(act_fn)

    if ada_comb:
        # Compute median of IOIs
        typ_dist = np.median(np.ediff1d(onsets))
        pp = OnsetPeakPickingProcessor(combine=typ_dist*0.5)
        onsets = pp.process_offline(act_fn)
    return onsets, typ_dist

def local_energy(file, fps, frame_size):
    sig = madmom.audio.signal.Signal(file, dtype=float)
    frames = madmom.audio.signal.FramedSignal(signal=sig, frame_size=frame_size, fps=fps)
    return frames.energy()

def legato_mg(file, rel_delta=None, sep=None):
    """
    (Maestre & Gómez, 2005)
    """
    FPS = 100 # Energy frame parameters
    WIN_SIZE = 1024
    onsets, typ_dist = bock_onsets(file)
    energy = local_energy(file, fps=FPS, frame_size=WIN_SIZE)
    # Compute energy at onset, before onset and after onset
    # Measure E_onset/min(E_before, E_after)
    # Separator is either given absolute or relative to median IOI.
    if sep is None:
        if rel_delta is None:
            sep = 0.2*typ_dist # default
        else:
            sep = rel_delta*typ_dist

    legato_meas = np.array([
        e0/np.min((e1,e2))
        for (e0, e1, e2) in
        zip(
            energy[(onsets*FPS).astype(int)],
            energy[((onsets - sep)*FPS).astype(int)],
            energy[((onsets + sep)*FPS).astype(int)]
        )
    ])
    
    # Output: Onset times together with legato measure
    return onsets, legato_meas

def onsets_threshold_gate(onsets, values, threshold):
    valid_idx = np.where(values>threshold)[0]
    return onsets[valid_idx]

if __name__=="__main__":
    onsets, leg = legato_mg(
            'datasets/initslurtest_vn/initslurtest_vn_wav/slurtest02.wav',
            0.2
    )
    print(onsets.shape)
    print(leg.shape)
    val_onsets = onsets_threshold_gate(onsets, leg, 1.0)
    print(val_onsets.shape)




    