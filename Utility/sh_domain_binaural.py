import numpy as np


def compute_auditory_cues_magLS(h_L, h_R, C, N, freq_window, tau_r):
    assert ((N + 1)**2 == h_L.shape[0])
    assert ((N + 1)**2 == h_R.shape[0])

    auto_spectrum_L = np.sum(np.conj(h_L) * (C @ h_L), axis=0)
    auto_spectrum_R = np.sum(np.conj(h_R) * (C @ h_R), axis=0)
    cross_spectrum = np.sum(np.conj(h_L) * (C @ h_R), axis=0)

    auto_spectrum_L = np.tile(auto_spectrum_L[np.newaxis, :],
                              (freq_window.shape[0], 1)) * freq_window
    auto_spectrum_R = np.tile(auto_spectrum_R[np.newaxis, :],
                              (freq_window.shape[0], 1)) * freq_window
    cross_spectrum = np.tile(cross_spectrum[np.newaxis, :],
                             (freq_window.shape[0], 1)) * freq_window

    P_L = np.fft.irfft(auto_spectrum_L, axis=1)[:, 0]
    P_R = np.fft.irfft(auto_spectrum_R, axis=1)[:, 0]

    ILD = 10 * np.log10(P_L / P_R)

    cross_correlation = np.fft.irfft(cross_spectrum, axis=1)
    IC = np.max(np.abs(cross_correlation)[:, tau_r], axis=1) / np.sqrt(
        P_L * P_R)

    # return mean of cues along frequency bands
    return IC, ILD
