import tensorflow as tf
import librosa
import utils
import numpy as np

def stft(x, n_fft=2048, n_hopsize=1024, center=True, window='hann', dtype=np.complex64):
    return librosa.core.stft(x, n_fft=n_fft, hop_length=n_hopsize, pad_mode='constant', center=center, window=window, dtype=dtype)


def istft(X, rate=44100, n_fft=2048, n_hopsize=1024, center=True, window='hann', dtype=np.float32):
    return librosa.core.istft(X, hop_length=n_hopsize, center=center, window=window, dtype=dtype)


def spectrogram(X, power):
    return np.abs(X)**power


if __name__ == "__main__":
    s = utils.sine()
    X = stft(s)
    x = istft(X, rate=44100)
    print(utils.rms(s, x))