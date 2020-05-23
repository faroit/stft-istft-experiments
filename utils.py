import numpy as np


def rms(x, y):
    return np.sqrt(np.mean((x - y)**2, axis=-1))


def sine(f=440, rate=44100, samples=1024*100, dtype=np.float32):
    return np.sin(2*np.pi*np.arange(samples) * f/rate).astype(dtype)
