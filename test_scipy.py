import numpy as np
import utils
import scipy.signal


def stft(x, n_fft=2048, n_hopsize=1024):
    f, t, X = scipy.signal.stft(
        x, 
        nperseg=n_fft, 
        noverlap=n_fft - n_hopsize, 
        padded=True,
    )
    return X * n_hopsize


def istft(X, rate=44100, n_fft=2048, n_hopsize=1024):
    t, audio = scipy.signal.istft(
        X / n_hopsize, 
        rate, 
        nperseg=n_fft, 
        noverlap=n_fft - n_hopsize, 
        boundary=True
    )
    return audio


def spectrogram(X, power):
    return np.abs(X)**power


if __name__ == "__main__":
    s = utils.sine()
    X = stft(s)
    x = istft(X, rate=44100)
    print(utils.rms(s, x))
