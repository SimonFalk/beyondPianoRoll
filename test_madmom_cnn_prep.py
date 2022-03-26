import matplotlib.pyplot as plt
import numpy as np

from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.filters import MelFilterbank
from madmom.audio.spectrogram import (FilteredSpectrogramProcessor,
                                    LogarithmicSpectrogramProcessor)
from madmom.processors import (BufferProcessor, OnlineProcessor, ParallelProcessor,
                          SequentialProcessor, )

def _cnn_onset_processor_pad(data):
    """Pad the data by repeating the first and last frame 7 times."""
    pad_start = np.repeat(data[:1], 7, axis=0)
    pad_stop = np.repeat(data[-1:], 7, axis=0)
    return np.concatenate((pad_start, data, pad_stop))

sig_proc = SignalProcessor(num_channels=1, sample_rate=44100)

multi_proc = ParallelProcessor([])
for frame_size in [2048, 1024, 4096]:
    frames_proc = FramedSignalProcessor(frame_size=4096, fps=100)

    stft_proc = ShortTimeFourierTransformProcessor()

    filt_proc = FilteredSpectrogramProcessor(
                    filterbank=MelFilterbank, num_bands=80, fmin=27.5, fmax=16000,
                    norm_filters=True, unique_filters=False)

    spec_proc = LogarithmicSpectrogramProcessor(log=np.log, add=np.spacing(1))

    multi_proc.append(SequentialProcessor((frames_proc, stft_proc, filt_proc, spec_proc)))

stack = np.dstack

# Pad removed

pre_processor = SequentialProcessor((sig_proc, multi_proc, stack))

prep = pre_processor("datasets/OnsetLabeledInstr2013/development/Violin/42954_FreqMan_hoochie_violin_pt1.wav")


np.save("results/processor_test/prep", prep)