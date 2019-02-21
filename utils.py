import numpy as np


def rms(x, y):
    return np.sqrt(np.mean((x - y)**2))


def sine(f=440, rate=44100, dur=4096*10):
    return np.sin(2*np.pi*np.arange(dur) * f/rate)
