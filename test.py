import test_scipy
import test_librosa
import test_torch
import utils
import numpy as np
import matplotlib.pyplot as plt

stfts = [test_torch, test_scipy, test_librosa]
istfts = [test_torch, test_scipy, test_librosa]

if __name__ == "__main__":
    s = utils.sine(dtype=np.float32)
    for forward_method in stfts:
        stft = getattr(forward_method, 'stft')
        for inverse_method in istfts:
            istft = getattr(inverse_method, 'istft')
            X = stft(s)
            x = istft(X)

            print(
                forward_method.__name__,
                "-->",
                inverse_method.__name__,
                utils.rms(s, x)
            )
