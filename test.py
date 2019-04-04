import test_scipy
import test_librosa
import test_torch
import test_tf
import utils
import numpy as np

stfts = [test_torch, test_scipy, test_librosa, test_tf]
istfts = [test_torch, test_scipy, test_librosa, test_tf]

n_fft = 2048
n_hopsize = 512

if __name__ == "__main__":
    s = utils.sine(dtype=np.float32)
    for forward_method in stfts:
        stft = getattr(forward_method, 'stft')
        for inverse_method in istfts:
            istft = getattr(inverse_method, 'istft')
            X = stft(s, n_fft=n_fft, n_hopsize=n_hopsize)
            x = istft(X, n_fft=n_fft, n_hopsize=n_hopsize)

            print(
                forward_method.__name__,
                "-->",
                inverse_method.__name__,
                utils.rms(s, x)
            )
