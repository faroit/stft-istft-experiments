import torch, torchaudio
import numpy as np
import utils

def stft(sig_vec, n_fft=None, n_hopsize=None, window=torch.hann_window, out_type="numpy", normalized=False):
    """ sig_vec = [batch, time]

        default values are consistent with librosa.core.spectrum._spectrogram:
        center = True,
        onesided = True,
        pad_mode = 'reflect'
    """

    if not (isinstance(sig_vec, torch.FloatTensor) or isinstance(sig_vec, torch.cuda.FloatTensor)):
        #print(sig_vec.shape)
        sig_vec = torch.from_numpy(sig_vec).float().view(-1, sig_vec.shape[-1]) # [batch, time]
    else:
        sig_vec = sig_vec.view(-1, sig_vec.shape[-1]) # [batch, time]
    #print(sig_vec.shape)

    if n_fft is None: n_fft = 2048 # better to be an even number ?
    if n_hopsize is None: n_hopsize = int(n_fft // 2)

    window_stft = window(n_fft, periodic=True)
    window_stft = window_stft.to(sig_vec.device)

    stft_mat = torch.stft(sig_vec,
        n_fft = n_fft,
        hop_length = n_hopsize,
        window = window_stft,
        center = True,
        normalized = normalized,
        onesided = True,
        pad_mode = 'reflect'
    )

    #print(stft_mat.shape)
    out_torch = torch.squeeze(stft_mat, dim=0)
    if out_type == "numpy":
        out_torch = out_torch.cpu().numpy()
        out_torch = out_torch[..., 0] + out_torch[..., 1]*1j # combine real and imaginary part
    #print(out_torch.shape)
    return out_torch


def istft(stft_mat, n_fft=None, n_hopsize=None, window=torch.hann_window, out_type="numpy", normalized=False):
    """ stft_mat = [batch, freq, time, complex]

        default values are consistent with librosa.core.spectrum._spectrogram:
        center = True,
        onesided = True,
        unpad_mode = 'reflect'
    """

    if not (isinstance(stft_mat, torch.FloatTensor) or isinstance(stft_mat, torch.cuda.FloatTensor)):
        #print(stft_mat.shape)
        stft_mat = torch.from_numpy(np.stack([np.real(stft_mat), np.imag(stft_mat)], axis=-1)).float().view((-1,) + stft_mat.shape[-2:] + (2,)) # [batch, freq, time, complex]
    else:
        stft_mat = stft_mat.view((-1,) + stft_mat.shape[-3:]) # [batch, freq, time, complex]
    #print(stft_mat.shape)

    if n_fft is None: n_fft = 2 * (stft_mat.shape[-3] - 1) # would always be an even number
    if n_hopsize is None: n_hopsize = int(n_fft // 2)

    window_istft = window(n_fft, periodic=True)
    window_istft = window_istft.to(stft_mat.device)

    sig_vec = torchaudio.functional.istft(stft_mat,
        n_fft = n_fft,
        hop_length = n_hopsize,
        window = window_istft,
        center = True,
        normalized = normalized,
        onesided = True,
        pad_mode = 'reflect'
    )

    #sig_vec = sig_vec / window_istft.sum()
    #print(sig_vec.shape)
    out_torch = torch.squeeze(sig_vec, dim=0)
    if out_type == "numpy":
        out_torch = out_torch.cpu().numpy()
    #print(out_torch.shape)
    return out_torch


def spectogram(X, power=1):
    return X.pow(2).sum(axis=-1).pow(power / 2.0)


if __name__ == "__main__":
    s = utils.sine()
#    s = np.stack([s, s, s, s])
    X = stft(s)
    x = istft(X)
#    print(s)
#    print(x)
    print(utils.rms(s, x))
