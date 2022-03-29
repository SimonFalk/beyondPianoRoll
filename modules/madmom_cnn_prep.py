import matplotlib.pyplot as plt
import numpy as np

from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.filters import MelFilterbank
from madmom.audio.spectrogram import (FilteredSpectrogramProcessor,
                                    LogarithmicSpectrogramProcessor)
from madmom.processors import (BufferProcessor, OnlineProcessor, ParallelProcessor,
                          SequentialProcessor, )

def cnn_preprocessor():
    def _cnn_onset_processor_pad(data):
        """Pad the data by repeating the first and last frame 7 times."""
        pad_start = np.repeat(data[:1], 7, axis=0)
        pad_stop = np.repeat(data[-1:], 7, axis=0)
        return np.concatenate((pad_start, data, pad_stop))

    EPSILON = np.spacing(1)

    sig = SignalProcessor(num_channels=1, sample_rate=44100)
    # process the multi-resolution spec in parallel
    multi = ParallelProcessor([])
    for frame_size in [2048, 1024, 4096]:
        frames = FramedSignalProcessor(frame_size=frame_size, fps=100)
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        filt = FilteredSpectrogramProcessor(
            filterbank=MelFilterbank, num_bands=80, fmin=27.5, fmax=16000,
            norm_filters=True, unique_filters=False)
        spec = LogarithmicSpectrogramProcessor(log=np.log, add=EPSILON)
        # process each frame size with spec and diff sequentially
        multi.append(SequentialProcessor((frames, stft, filt, spec)))
    # stack the features (in depth) and pad at beginning and end
    stack = np.dstack
    pad = _cnn_onset_processor_pad
    # pre-processes everything sequentially
    pre_processor = SequentialProcessor((sig, multi, stack, pad))
    return pre_processor
