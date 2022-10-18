import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def onset_visualizer(passage=None, audio=None, onset_list=None, lims=None, onset_styles=None, ax=None, **plt_kwargs):
    """
    Either use a passage as created from passage_extractor()
    or manually select the audio, onsetlist and 
    limits (only for visualization purpose)
    """
    
    #HEIGHT_SEP = 0.2*np.max(audio)
    HEIGHT_SEP=0

    if passage is not None:
        audio = passage["audio"]
        onset_list = passage["onsets"]
        lims = (passage["abs_start"], passage["abs_end"])
    if onset_styles is None:
        ONSET_MARKERS = ["v", "^", "^"]
        ONSET_COLORS = ["k", "r", "g"]
    else:
        ONSET_MARKERS = onset_styles["m"]
        ONSET_COLORS = onset_styles["c"]
    if ax is None:
        ax = plt.gca()
    o = 0
    ax.plot(np.linspace(lims[0], lims[1], len(audio)), audio, zorder=0, **plt_kwargs)

    while o<len(onset_list):
        if o<len(ONSET_MARKERS) and o<len(ONSET_COLORS):
            ax.scatter(onset_list[o], np.zeros_like(onset_list[o])-o*HEIGHT_SEP, 
                    marker=ONSET_MARKERS[o], c=ONSET_COLORS[o], s=30.0, zorder=len(onset_list)+10-o
            )
        else:
            ax.scatter(onset_list[o], np.zeros_like(onset_list[o])-o*HEIGHT_SEP, s=30.0),
        o+=1
    return ax

def slur_visualizer(onsets, slur_onehot, sign=1, slur_height=None, edgecolor=None, ax=None): 
    if slur_height is None:
        SLUR_HEIGHT = 7000
    else:
        SLUR_HEIGHT = slur_height
    if ax is None:
        ax = plt.gca()

    starts = onsets*slur_onehot[:,0]
    ends = onsets[1:]*slur_onehot[1:,2] # first onset can't be an end
    starts = starts[starts!=0]
    ends = ends[ends!=0]
    Path = mpath.Path
    pps = []
    for n in np.arange(0, min(len(starts), len(ends))):
        onset = starts[n]
        next_onset = ends[n]
        pps.append(mpatches.PathPatch(
            Path([(onset, sign*SLUR_HEIGHT*0.25), (.5*(onset+next_onset), sign*SLUR_HEIGHT), (next_onset, sign*SLUR_HEIGHT*0.25)],
            [Path.MOVETO, Path.CURVE3, Path.CURVE3]),
            edgecolor=edgecolor, transform=ax.transData))
    [ax.add_patch(pp) for pp in pps]
    



def passage_extractor(audio, onset_list, breakpoints, sr=44100):
    """
    Extract a list of passages given audio and channels of annotations.
    Breakpoints are given as [p0, p1, p2] where pn are the points to break up the sequence.
    """
    return [{
            "audio" : audio[int(start*sr):int(end*sr)],
            "onsets": [onsets[np.intersect1d(
                np.where(onsets-start>0),
                np.where(onsets-end<0)
            )] for onsets in onset_list],
            "abs_start" : start,
            "abs_end" : end
        } for start, end in zip(breakpoints[:-1], breakpoints[1:])]