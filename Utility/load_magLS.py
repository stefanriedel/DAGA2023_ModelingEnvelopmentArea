import soundfile
import numpy as np
import signal


def binaural_decode(x, fs, hrir_path):
    hrir, fs_binaural = soundfile.read(hrir_path)
    assert fs == fs_binaural

    # compute multiplier for left/right multiplier
    nm = np.arange(0, hrir.shape[1])
    n = np.floor(np.sqrt(nm))
    m = nm - n**2 - n
    mult = np.zeros(hrir.shape[1])
    mult[m >= 0] = 1
    mult[m < 0] = -1

    left_sh = signal.fftconvolve(x, hrir, axes=[
        0,
    ])
    left_sh = left_sh[:x.shape[0], :]
    right = np.sum(left_sh * mult[None, :], axis=-1)
    left = np.sum(left_sh, axis=-1)

    return np.stack([left, right], axis=-1)


def load_magLS_decoder(hrir_path):
    hrir, fs_binaural = soundfile.read(hrir_path)

    # compute multiplier for left/right multiplier
    nm = np.arange(0, hrir.shape[1])
    n = np.floor(np.sqrt(nm))
    m = nm - n**2 - n
    mult = np.zeros(hrir.shape[1])
    mult[m >= 0] = 1
    mult[m < 0] = -1

    right_sh = hrir * mult[None, :]
    left_sh = hrir

    return np.stack([left_sh, right_sh], axis=-1)
