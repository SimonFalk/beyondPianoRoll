import numpy as np
import madmom
from madmom.features.onsets import CNNOnsetProcessor, OnsetPeakPickingProcessor

def simple_energy_onsets(file, base_onsets=None, energy_thres=1.0, onset_thres=0.7, combine=0.1, frame_size=1024, fps=100):
    if base_onsets is None:
        cnn = CNNOnsetProcessor(fps=fps)
        act_fn = cnn(file)
        pp = OnsetPeakPickingProcessor(threshold=onset_thres, combine=combine, fps=fps)
        onsets = pp.process_offline(act_fn)
    else:
        onsets = base_onsets

    sig = madmom.audio.signal.Signal(file, dtype=float)
    frames = madmom.audio.signal.FramedSignal(signal=sig, frame_size=frame_size, fps=fps)
    energy_at_onsets = frames.energy()[(onsets*fps).astype(int)]

    hard_idx = np.where(energy_at_onsets<energy_thres)[0]
    soft_idx = np.where(energy_at_onsets>energy_thres)[0]

    return onsets[hard_idx], onsets[soft_idx]

    

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

if __name__=="__main__":
    onsets, leg = legato_mg(
            'datasets/initslurtest_vn/initslurtest_vn_wav/slurtest02.wav',
            0.2
    )
    print(onsets.shape)
    print(leg.shape)
    val_onsets = onsets_threshold_gate(onsets, leg, 1.0)
    print(val_onsets.shape)




    