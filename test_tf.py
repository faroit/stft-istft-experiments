import tensorflow as tf
import numpy as np
import utils
import functools
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.enable_eager_execution()

def stft(x, n_fft=2048, n_hopsize=1024, center=True, out_type="numpy"):
    """Code from: 
    https://github.com/tensorflow/tensorflow/issues/16465#issuecomment-396494851
    """
    audio = tf.cast(x, tf.float32)
    if center:
        # librosa pads by n_fft, which almost works perfectly here, except for with frame_step 256.
        pad_amount = 2 * (n_fft - n_hopsize)
        audio = tf.pad(audio, [[pad_amount // 2, pad_amount // 2]], 'CONSTANT')

    f = tf.contrib.signal.frame(audio, n_fft, n_hopsize, pad_end=False)
    w = tf.contrib.signal.hann_window(n_fft, periodic=True)
    stft_tf = tf.transpose(tf.spectral.rfft(f * w, fft_length=[n_fft]))

    if out_type == "tf":
        return stft_tf
    elif out_type == "numpy":
        return stft_tf.numpy()



def istft(X, n_fft=2048, n_hopsize=1024, center=True):
    X = tf.cast(X, tf.complex64)
    pad_amount = 2 * (n_fft - n_hopsize)
    audio_tf = tf.contrib.signal.inverse_stft(
        tf.transpose(X), n_fft, n_hopsize,
        window_fn=tf.contrib.signal.inverse_stft_window_fn(n_hopsize))
    if center and pad_amount > 0:
        audio_tf = audio_tf[pad_amount // 2:-pad_amount // 2]

    return audio_tf


def spectrogram(X, power):
    return tf.abs(X)**power


if __name__ == "__main__":
    s = utils.sine()
    X = stft(s)
    x = istft(X)
    print(utils.rms(s, x))

