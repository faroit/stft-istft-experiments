import test_scipy
import test_librosa
import test_torch
import utils
import numpy as np

stfts = [test_torch, test_scipy, test_librosa, test_tf]
istfts = [test_torch, test_scipy, test_librosa, test_tf]

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
