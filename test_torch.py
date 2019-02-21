import torch
import numpy as np
import utils


def stft(sig, n_fft=2048, hop_length=1024, window=torch.hann_window, out_type="numpy"):
    if not isinstance(sig, torch.DoubleTensor):
        sig_t = torch.from_numpy(np.atleast_2d(sig)).float()
    else:
        sig_t = sig

    window_t = window(n_fft)
    window_t.to(sig_t.device)

    # default values are consistent with librosa.core.spectrum._spectrogram
    stft_f = torch.stft(sig_t, n_fft=n_fft, hop_length=hop_length,
        window=window_t, center=True,
        normalized=False, onesided=True,
        pad_mode='reflect'
    ).transpose(1, 2)

    out_torch = stft_f.squeeze().cpu().numpy().T

    if out_type == "torch":
        return out_torch
    elif out_type == "numpy":
        # combine real and imaginary part
        return out_torch[0, ...] + out_torch[1, ...]*1j


def spectogram(X, power=1):
    return X.pow(2).sum(axis=-1).pow(power / 2.0)


if __name__ == "__main__":
    s = utils.sine()
    X = stft(s)
    print(X.shape)
