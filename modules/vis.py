import numpy as np

def onset_visualizer(audio, onset_list, lims, ax=None, **plt_kwargs):
    ONSET_MARKERS = ["v", "^", "^"]
    ONSET_COLORS = ["k", "r", "g"]
    if ax is None:
        ax = plt.gca()
    o = 0
    ax.plot(np.linspace(lims[0], lims[1], len(audio)), audio, zorder=0, **plt_kwargs)

    while o<len(onset_list):
        if o<len(ONSET_MARKERS) and o<len(ONSET_COLORS):
            ax.scatter(onset_list[o], np.zeros_like(onset_list[o]), 
                    marker=ONSET_MARKERS[o], c=ONSET_COLORS[o], zorder=len(onset_list)+10-o
            )
        else:
            ax.scatter(onset_list[o], np.zeros_like(onset_list[o])),
        o+=1
    return ax


def passage_extractor(audio, onset_list, breakpoints, sr=44100):
    return [{
            "audio" : audio[int(start*sr):int(end*sr)],
            "onsets": [onsets[np.intersect1d(
                np.where(onsets-start>0),
                np.where(onsets-end<0)
            )] for onsets in onset_list],
            "abs_start" : start,
            "abs_end" : end
        } for start, end in zip(breakpoints[:-1], breakpoints[1:])]