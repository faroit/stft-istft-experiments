import torch
import numpy as np
import utils

def stft(sig_vec, n_fft=None, hop_length=None, window=torch.hann_window, out_type="numpy"):
    """ sig_vec = [batch, time]

        default values are consistent with librosa.core.spectrum._spectrogram:
        center = True,
        normalized = False,
        onesided = True,
        pad_mode = 'reflect'
    """

    if not isinstance(sig_vec, torch.DoubleTensor):
        sig_vec = torch.from_numpy(np.atleast_2d(sig_vec)).float()
    # sig_vec = sig_vec.to('cuda')

    if n_fft is None: n_fft = 2048 # better to be an even number ?
    if hop_length is None: hop_length = int(n_fft // 2)

    window_stft = window(n_fft)
    window_stft = window_stft.to(sig_vec.device)

    stft_mat = torch.stft(sig_vec,
        n_fft = n_fft,
        hop_length = hop_length,
        window = window_stft,
        center = True,
        normalized = False,
        onesided = True,
        pad_mode = 'reflect'
    ).transpose(1, 2)

    out_torch = stft_mat.squeeze().cpu().numpy().T
    if out_type == "torch":
        return out_torch
    elif out_type == "numpy":
        return out_torch[0, ...] + out_torch[1, ...]*1j # combine real and imaginary part

def istft(stft_mat, hop_length=None, window=torch.hann_window):
    """ stft_mat = [batch, freq, time, complex]

        default values are consistent with librosa.core.spectrum._spectrogram:
        center = True,
        normalized = False,
        onesided = True,
        unpad_mode = 'reflect'

    All based on librosa
        - http://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#istft
    What's missing?
        - normalize by sum of squared window --> do we need it here?
        Actually the result is ok by simply dividing y by 2. 
    """

    if not isinstance(stft_mat, torch.DoubleTensor):
        stft_mat = torch.from_numpy(np.stack([np.real(stft_mat), np.imag(stft_mat)]).transpose((1, 2, 0))[None, ...]).float()
    # stft_mat = stft_mat.to('cuda')

    n_fft = 2 * (stft_mat.shape[-3] - 1) # would always be an even number
    if hop_length is None: hop_length = int(n_fft // 2)

    window_istft = window(n_fft)
    window_istft = window_istft.to(stft_mat.device)

    n_frames = stft_mat.shape[-2] # [time] (time domain of stft_mat)
    n_samples = n_fft + hop_length * (n_frames - 1) # [time] (time domain of reconstructed signal)
    window_istft = window_istft.view(1, -1) # [batch, time]
    sig_vec = torch.zeros(stft_mat.shape[0], n_samples, device=stft_mat.device) # [batch, time]
    win_vec = torch.zeros(stft_mat.shape[0], n_samples, device=stft_mat.device) # [batch, time]
    win_vec_1frame = window_istft ** 2 # [batch, time]
    for i in range(n_frames):
        sig_vec_1frame = torch.irfft(stft_mat[:, :, i], signal_ndim=1, signal_sizes=(n_fft,)) # [batch, time]
        sig_vec_1frame *= window_istft # [batch, time]

        idx_sig = i * hop_length
        sig_vec[:, idx_sig:(idx_sig+n_fft)] += sig_vec_1frame
        win_vec[:, idx_sig:(idx_sig+n_fft)] += win_vec_1frame
    sig_vec /= win_vec
    sig_vec = sig_vec[:, n_fft//2:-n_fft//2] # unpadding

    # out_torch = (sig_vec / window_istft.sum()).squeeze().cpu().numpy()
    out_torch = (sig_vec).squeeze().cpu().numpy()
    return out_torch


def spectogram(X, power=1):
    return X.pow(2).sum(axis=-1).pow(power / 2.0)


if __name__ == "__main__":
    s = utils.sine()
    X = stft(s)
    x = istft(X)
    print(utils.rms(s, x))
