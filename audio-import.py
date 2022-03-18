import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt


audio = tfio.audio.AudioIOTensor('../DoReMir/initslurtest_vn/initslurtest_vn_wav/slurtest04.wav')
audio_tensor = tf.squeeze(audio, axis=[-1])
print(audio_tensor.shape)
spectrogram = tfio.audio.spectrogram(audio_tensor, nfft=1024, window=1024, stride=512)
