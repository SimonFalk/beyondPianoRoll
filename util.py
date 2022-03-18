from pydub import AudioSegment
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

def compress_log(values, gamma=100):
    '''values is a np array, 
    gamma is the compression constant'''
    return np.log(np.ones(values.shape) + gamma*values)


def compute_scale_features(stft, sampling_rate, n_sample_pts):
    N, K = stft.shape
    print(N, K)
    
    def phys_freq(coef):
        return coef*sampling_rate/n_sample_pts
    
    def center_freq(p):
        return 440*2**((p-69)/12)
    P_upper = [center_freq(p+0.5) for p in range(128)]

    log_freq = np.zeros((N,128))
    chroma = np.zeros((N,12))
    
    for time_frame in range(N):
        # Sum up squared magnitudes of coefficients belonging to pitch class
        ind_to_pitch = np.digitize(phys_freq(np.arange(K)), P_upper)
        for fourier_index in range(K):
            pitch = ind_to_pitch[fourier_index] if ind_to_pitch[fourier_index]<128 else 127
            log_freq[time_frame][pitch] += np.abs(stft[time_frame][fourier_index])**2
            chroma[time_frame][pitch%12] += np.abs(stft[time_frame][fourier_index])**2
    
    return [log_freq, chroma]

class MirAudioSegment(AudioSegment):
    def __init__(self, data=None, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.stft = None
        
    def is_mono(self):
        return self.channels == 1
    
    def samples(self):
        if self.is_mono():
            return super().get_array_of_samples()
        else:
            return super().get_array_of_samples()[::2]
        
    def get_timeline(self,offset=0):
        N = int(self.frame_rate * self.duration_seconds)
        return np.linspace(offset,offset+self.duration_seconds,N)
    
    def plot_waveform(self, offset=0):
        ax = plt.axes()
        ax.set_xlabel('Time [seconds]')
        ax.set_ylabel('Amplitude')
        ax.plot(
                self.get_timeline(offset),
                self.samples()
        )
        return ax
        
    def compute_stft(self, w_length, hop_factor=0.5, window="hann"):
        """
        w_length is window size in physical time
        hop factor is hop size/window size
        """
        self.stft_w_length = w_length
        self.stft = signal.stft(
                                self.samples(),
                                fs=self.frame_rate,
                                window=window,
                                nperseg=int(self.frame_rate*w_length),
                                noverlap=int(self.frame_rate*w_length*(1-hop_factor)),
        )
        return self.stft 
    
    def plot_magnitude(self, squared=True):
        ax = plt.axes()
        f, t, coeff = self.stft
        if squared:
            z = np.abs(coeff)**2
        else:
            z = np.abs(coeff)
        ax.pcolormesh(t, f, z, vmin=0, cmap='inferno', shading='auto')
        ax.set_title('STFT magnitude spectrum')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [s]')
        ax.set_ylim([0,self.frame_rate*self.stft_w_length//2])
        return ax
    
    def log_freq_spectrogram(self, w_length=None, hop_factor=None):
        if not self.stft:
            print("Must compute STFT first!")
            return
        [lf, chroma] = compute_scale_features(self.stft[2], self.frame_rate, len(self.samples()))
        self.log_freq = lf
        return lf
    
    def plot_lf_spectrogram(self):
        ax = plt.axes()
        lf = self.log_freq
        N, P = lf.shape
        ax.pcolormesh(np.linspace(0, self.duration_seconds, N),
                        np.arange(P), 
                        lf.T, 
                        vmin=0, 
                        cmap="inferno",
                        shading='auto')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("MIDI pitch")
        ax.set_ylim([50,120])
        return ax
    
    def plot_chroma(self, gamma=None):
        ax = plt.axes()
        chroma = self.chroma
        if gamma is not None:    
            chroma = np.log(np.ones(chroma.shape) + gamma*chroma)
        N, C = chroma.shape
        ax.pcolormesh(np.linspace(0, self.duration_seconds, N),
                        np.arange(C), 
                        chroma.T, 
                        vmin=0, shading='auto')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Note")
        return ax