import numpy as np
from os.path import dirname, join as pjoin
import os
from joblib import Parallel, delayed
from Utility.load_magLS import load_magLS_decoder
from Utility.sh_domain_binaural import compute_auditory_cues_magLS
from Utility.ambisonics import sph2cart, sh_xyz


def computeExponent(spl_decay_distancedoubling):
    return np.log2(10**(spl_decay_distancedoubling / 20))


def computeDirectivityFactor(directivity_index):
    return 10**(directivity_index / 10)


def computeDirectivityIndex(directivity_factor):
    return 10 * np.log10(directivity_factor)


def computeCardioidDirectivityWeights(axis_xyz, eval_xyz):
    weights = np.zeros(axis_xyz.shape[0])
    for idx in range(axis_xyz.shape[0]):
        axis_xyz[idx, :] /= np.linalg.norm(axis_xyz[idx, :])
        eval_xyz[idx, :] /= np.linalg.norm(eval_xyz[idx, :])
        weights[idx] = 0.5 + 0.5 * np.dot(axis_xyz[idx, :], eval_xyz[idx, :])

    return weights


def computeSHCovarianceMatrix(src_dirs, g, N, room_volume, room_T60,
                              directivity_index):
    Y = sh_xyz(N, src_dirs[:, 0], src_dirs[:, 1], src_dirs[:, 2]).transpose()
    diag_g_sqrd = np.diag(g**2)
    YGY = Y @ diag_g_sqrd @ Y.transpose()

    eps = 1e-6
    room_T60 = room_T60 + eps
    gamma = computeDirectivityFactor(directivity_index)
    r_H = 0.057 * np.sqrt(room_volume / room_T60) * np.sqrt(gamma)

    C = YGY + np.eye(YGY.shape[0]) * g.size / (r_H**2 * 4 * np.pi)
    return C


def computeAndSaveCues(ls_xyz,
                       source_exponents,
                       config_names,
                       head_rotations,
                       room_volume,
                       room_T60_list,
                       directivity_index_list,
                       IC_range,
                       ILD_range,
                       area_len,
                       save_path,
                       hrir_path,
                       N_ambi=1,
                       max_cue=False,
                       dataset_name=''):

    root_dir = dirname(__file__)
    utility_path = pjoin(root_dir, 'Utility')

    # Define meshgrid simulation area
    # array_radius = 5
    orig_area_len = area_len
    area_len = area_len + 0.1 * area_len  # array_radius + 0.1 * array_radius
    x_ls, y_ls, z_ls = ls_xyz[:, 0], ls_xyz[:, 1], ls_xyz[:, 2]

    # Define source signal covariance matrix, default is unit matrix
    Cov = np.eye(x_ls.size)
    # Cov = np.ones((phi_ls.size,phi_ls.size))

    # Stack source coordinates
    ls = np.array([x_ls, y_ls, z_ls]).transpose()

    # Define source model list
    num_source_models = source_exponents.shape[0]
    assert (len(config_names) == num_source_models)
    assert (len(config_names) == len(directivity_index_list))
    assert (len(config_names) == len(room_T60_list))
    num_variables = len(config_names)

    # Create listener meshgrid
    x = np.arange(-area_len, area_len + 0.05 * orig_area_len,
                  0.05 * orig_area_len)
    y = np.arange(-area_len, area_len + 0.05 * orig_area_len,
                  0.05 * orig_area_len)
    res = x.size

    [list_X, list_Y] = np.meshgrid(x, y)
    listener = np.vstack((list_X.flatten(), list_Y.flatten(),
                          np.zeros(list_Y.flatten().size))).transpose()

    # Load gammatone magnitude windows, precomputed using the 'pyfilterbank' library
    # https://github.com/SiggiGue/pyfilterbank
    filename = 'gammatone_erb_mag_windows_nfft_1024_numbands_320.npy'
    gammatone_mag_win = np.load(pjoin(utility_path, filename))
    Nfft = int((gammatone_mag_win.shape[1] - 1) * 2)
    num_bands = gammatone_mag_win.shape[0]
    filename = 'gammatone_fc_numbands_320_fs_48000.npy'
    f_c = np.load(pjoin(utility_path, filename))

    # ERB scale requires using every 8th of the 320 windows
    f_c = f_c[::8]
    gammatone_mag_win = gammatone_mag_win[::8]

    # Load MagLS HRTF set
    N_ambi = int(N_ambi)
    assert (N_ambi <= 7 and N_ambi >= 1)
    path_to_hrir = os.path.join(hrir_path, 'irsOrd' + str(N_ambi) + '.wav')
    hrir = load_magLS_decoder(path_to_hrir)
    N = int(np.sqrt(hrir.shape[1] - 1))
    fs = 48000
    hrtf = np.fft.rfft(hrir, n=Nfft, axis=0)
    h_L = hrtf[:, :, 0].transpose()
    h_R = hrtf[:, :, 1].transpose()

    # Define head rotations used for evaluation of IC and ILD
    rotations = head_rotations

    num_rotations = len(rotations)
    num_listener_pos = listener.shape[0]
    num_speaker_pos = ls.shape[0]

    theta = np.zeros((num_speaker_pos, 3, num_listener_pos))

    ls = np.tile(ls[:, :, np.newaxis], (1, 1, num_listener_pos))
    listener = listener.transpose()
    listener = np.tile(listener[np.newaxis, :, :], (num_speaker_pos, 1, 1))

    theta = ls - listener

    r = np.linalg.norm(theta, axis=1)
    r = np.tile(r[:, np.newaxis, :], (1, 3, 1))

    theta_norm = theta / r
    r_norm = r
    r_norm = r_norm[:, 0, :]

    source_phi = np.arctan2(theta_norm[:, 1, :],
                            theta_norm[:, 0, :]) / np.pi * 180.0  # + 90.0
    source_phi = source_phi.astype(np.int32)

    source_zen = np.arccos(theta_norm[:, 2, :])

    LEV_ILD = np.zeros((num_variables, num_listener_pos, num_rotations))
    LEV_IC = np.zeros((num_variables, num_listener_pos, num_rotations))
    BAL = np.zeros((num_variables, num_listener_pos, num_rotations))

    IC_low_lim = int(np.where(f_c >= IC_range[0])[0][0])
    IC_up_lim = int(np.where(f_c >= IC_range[1])[0][0])

    ILD_low_lim = int(np.where(f_c >= ILD_range[0])[0][0])
    ILD_up_lim = int(np.where(f_c >= ILD_range[1])[0][0])
    freq_window = gammatone_mag_win

    # tau search range, typically +- 1 millisecond
    tau_r = np.arange(-int(fs / 1000), int(fs / 1000))

    def mainLoopSourceModels(s):
        LEV_IC_tmp = np.zeros((num_listener_pos, num_rotations))
        LEV_ILD_tmp = np.zeros((num_listener_pos, num_rotations))
        BAL_tmp = np.zeros((num_listener_pos, num_rotations))

        w_r = np.zeros((x_ls.size, num_listener_pos))
        exponents = source_exponents[s, :]
        directivity_index = directivity_index_list[s]
        room_T60 = room_T60_list[s]

        for i in range(x_ls.size):
            w_r[i, :] = r_norm[i, :]**exponents[i]

        for rot in range(num_rotations):
            for p in range(num_listener_pos):
                azi = source_phi[:, p] + rotations[rot]
                azi = azi / 180 * np.pi
                zen = source_zen[:, p]
                src_dirs = sph2cart(azi, zen).transpose()

                # We could apply some directivity weights here
                # rot_src_dirs = sph2cart(azi - np.pi / 2, zen).transpose()
                # w_Gamma = computeCardioidDirectivityWeights(ls_xyz, rot_src_dirs)
                g = w_r[:, p]  # * w_Gamma

                # We assume uncorrelated signals of unit variances
                C = computeSHCovarianceMatrix(src_dirs, g, N, room_volume,
                                              room_T60, directivity_index)
                IC, ILD = compute_auditory_cues_magLS(h_L, h_R, C, N,
                                                      freq_window, tau_r)

                # Average across freq ranges:
                IC = np.mean(IC[IC_low_lim:IC_up_lim])
                ILD = np.mean(ILD[ILD_low_lim:ILD_up_lim])

                # Careful: LEV_IC holds 1-IC values for direct plotting
                LEV_IC_tmp[p, rot] = 1 - IC
                LEV_ILD_tmp[p, rot] = -np.abs(ILD)
                BAL_tmp[p, rot] = 20 * np.log10(
                    np.max(w_r[:, p]) / np.min(w_r[:, p]))

        return LEV_IC_tmp, LEV_ILD_tmp, BAL_tmp

    print('Main simulation loop started... \n')
    result_lists = Parallel(n_jobs=3)(delayed(mainLoopSourceModels)(s)
                                      for s in range(num_variables))
    print('Data is saved now.')

    res_arr = np.asarray(result_lists)

    for s in range(num_variables):
        LEV_IC[s, :, :] = res_arr[s, 0, :, :]
        LEV_ILD[s, :, :] = res_arr[s, 1, :, :]
        BAL[s, :, :] = res_arr[s, 2, :, :]

    if max_cue:
        LEV_ILD = np.min(LEV_ILD, axis=-1)
        LEV_IC = np.min(LEV_IC, axis=-1)
        BAL = -np.max(BAL, axis=-1)
    else:
        LEV_ILD = np.mean(LEV_ILD, axis=-1)
        LEV_IC = np.mean(LEV_IC, axis=-1)
        BAL = -np.mean(BAL, axis=-1)

    LEV_ILD = np.reshape(LEV_ILD, (num_variables, res, res))
    LEV_IC = np.reshape(LEV_IC, (num_variables, res, res))
    BAL = np.reshape(BAL, (num_variables, res, res))

    data_to_save = {
        'LEV_ILD': LEV_ILD,
        'LEV_IC': LEV_IC,
        'BAL': BAL,
        'list_X': list_X,
        'list_Y': list_Y,
        'ls_xyz': ls_xyz,
        'variable_names': config_names
    }

    np.save(pjoin(save_path, 'Data_' + dataset_name),
            data_to_save,
            allow_pickle=True)
