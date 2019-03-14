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
        window=window_t, 
        center=True,
        normalized=False, 
        onesided=True,
        pad_mode='reflect'
    ).transpose(1, 2)

    out_torch = stft_f.squeeze().cpu().numpy().T

    if out_type == "torch":
        return out_torch
    elif out_type == "numpy":
        # combine real and imaginary part
        return out_torch[0, ...] + out_torch[1, ...]*1j



def istft(stft_matrix, hop_length=1024, win_length=2048, window=torch.hann_window,
          center=True, normalized=False, onesided=True, length=None, out_type="numpy"):
    """stft_matrix = (batch, freq, time, complex) 

    # following Keunwoo Choi's implementation of istft.
    # https://gist.github.com/keunwoochoi/2f349e72cc941f6f10d4adf9b0d3f37e#file-istft-torch-py
    
    All based on librosa
        - http://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#istft
    What's missing?
        - normalize by sum of squared window --> do we need it here?
        Actually the result is ok by simply dividing y by 2. 
    """
    assert normalized == False
    assert onesided == True
    assert center == True

    if not isinstance(stft_matrix, torch.DoubleTensor):
        stacked_t = np.stack([np.real(stft_matrix), np.imag(stft_matrix)]).transpose((1, 2, 0))[None, ...]
        stft_matrix = torch.from_numpy(stacked_t).float()
    else:
        stft_matrix = stft_matrix

    device = stft_matrix.device
    n_fft = 2 * (stft_matrix.shape[-3] - 1)

    batch = stft_matrix.shape[0]

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = int(win_length // 4)

    istft_window = window(n_fft)
    istft_window.to(istft_window.device).view(1, -1)  # (batch, freq)

    n_frames = stft_matrix.shape[-2]
    expected_signal_len = n_fft + hop_length * (n_frames - 1)
    
    y = torch.zeros(batch, expected_signal_len, device=device)
    for i in range(n_frames):
        sample = i * hop_length
        spec = stft_matrix[:, :, i]
        iffted = torch.irfft(spec, signal_ndim=1, signal_sizes=(win_length,))

        ytmp = istft_window *  iffted
        y[:, sample:(sample+n_fft)] += ytmp
    
    # undo padding
    y = y[:, n_fft//2:-n_fft//2]
    
    if length is not None:
        if y.shape[1] > length:
            y = y[:, :length]
        elif y.shape[1] < length:
            y = torch.cat(y[:, :length], torch.zeros(y.shape[0], length - y.shape[1], device=y.device))
    
    coeff = n_fft/float(hop_length) / 2.0  # -> this might go wrong if curretnly asserted values (especially, `normalized`) changes.
    
    if out_type == "torch":
        return y / coeff
    elif out_type == "numpy":
        return np.squeeze(y / coeff).numpy()



def spectogram(X, power=1):
    return X.pow(2).sum(axis=-1).pow(power / 2.0)


if __name__ == "__main__":
    s = utils.sine()
    X = stft(s)
    print(X.shape)
