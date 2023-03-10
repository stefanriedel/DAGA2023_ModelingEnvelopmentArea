import numpy as np
from computeAndSaveCues import computeExponent
from Utility.ambisonics import sph2cart
from Utility.LoudspeakerArray import LoudspeakerArray


def zen(num_sources, elevation_angle):
    return np.pi / 2 - np.ones(num_sources) * elevation_angle / 180 * np.pi


def generateParameterList():
    # This script creates a list of parameters to render all plots of DAGA2023 paper

    # Overall number of plots
    num_plots = 11

    # Head rotations for simulation
    # head_rotations = [0, 30, 60, 90, 120, 150, -180, -150, -120, -90, -60, -30]
    head_rotations = [90]
    head_rotations = [head_rotations] * num_plots

    # Loudspeaker coordinates on unit sphere
    nLS1 = 12
    nLS2 = 8
    nLS3 = 4
    azi1 = (np.linspace(90, -270 + (360 / nLS1), nLS1) - 30) / 180 * np.pi
    azi2 = np.linspace(90 - 22.5, -270 - 22.5 +
                       (360 / nLS2), nLS2) / 180 * np.pi
    azi3 = np.linspace(90, -270 + (360 / nLS3), nLS3) / 180 * np.pi
    stereo_azi = np.array([45, 135]) / 180 * np.pi
    pm90_azi = np.array([0, 180]) / 180 * np.pi
    m90_azi = np.array([180]) / 180 * np.pi

    array_radius = 5
    ls_xyz1 = sph2cart(azi1, zen(nLS1, 0)).transpose() * array_radius
    ls_xyz2 = sph2cart(azi1, zen(nLS1, 15)).transpose() * array_radius
    ls_xyz3 = sph2cart(azi1, zen(nLS1, 30)).transpose() * array_radius
    ls_xyz4 = sph2cart(np.hstack((azi1, azi2, azi3)),
                       np.hstack((zen(nLS1, 0), zen(nLS2, 30), zen(
                           nLS3, 60)))).transpose() * array_radius
    ls_stereo = sph2cart(stereo_azi, zen(2, 0)).transpose() * array_radius
    m90_azi = sph2cart(m90_azi, zen(1, 0)).transpose()

    cube_coord = LoudspeakerArray('Cube').getCoord() / 180.0 * np.pi
    ls_cube = sph2cart(cube_coord[:, 0] + np.pi / 2,
                       np.pi / 2 - cube_coord[:, 1]).transpose()

    cube_dist = np.array([
        4.7, 5.0, 6.1, 5.2, 5.6, 5.6, 4.4, 5.6, 5.6, 5.2, 6.1, 5.0, 5.4, 5.6,
        6.1, 4.5, 4.6, 6.1, 5.4, 5.4
    ])
    ls_cube = ls_cube[:20, :] * np.tile(cube_dist[:, np.newaxis], (1, 3))

    ls_xyz = [
        ls_xyz1, ls_xyz2, ls_xyz3, ls_xyz4, ls_xyz1, ls_xyz1, ls_xyz1,
        ls_stereo, ls_cube[:12, :], ls_cube[:12, :], ls_cube[12:20, :]
    ]

    # Source directivity index
    DI0 = [0, 0, 0]
    DI4p7 = [4.77, 4.77, 4.77]
    DI8 = [8, 8, 8]
    DI10 = [10, 10, 10]
    directivity_index = [
        DI0, DI0, DI0, DI0, DI0, DI4p7, DI8, DI4p7, DI0, DI8, DI8
    ]

    # Room T60
    T60_0 = [0, 0, 0]
    T60_var = [0, 0.5, 1.0]

    room_T60 = [
        T60_0, T60_0, T60_0, T60_0, T60_var, T60_var, T60_var, T60_var,
        T60_var, T60_var, T60_var
    ]

    num_src = 12
    b1 = np.array([
        np.ones(num_src) * computeExponent(-1.5),
        np.ones(num_src) * computeExponent(-3),
        np.ones(num_src) * computeExponent(-6)
    ])
    num_src = 24
    b2 = np.array([
        np.ones(num_src) * computeExponent(-3),
        np.concatenate((np.ones(12) * computeExponent(-3),
                        np.ones(12) * computeExponent(-6))),
        np.ones(num_src) * computeExponent(-6)
    ])
    num_src = 12
    b3 = np.array([
        np.ones(num_src) * computeExponent(-6),
        np.ones(num_src) * computeExponent(-6),
        np.ones(num_src) * computeExponent(-6)
    ])
    num_src = 2
    b4 = np.array([
        np.ones(num_src) * computeExponent(-6),
        np.ones(num_src) * computeExponent(-6),
        np.ones(num_src) * computeExponent(-6)
    ])
    num_src = 8
    b5 = np.array([
        np.ones(num_src) * computeExponent(-6),
        np.ones(num_src) * computeExponent(-6),
        np.ones(num_src) * computeExponent(-6)
    ])

    source_exponents = [b1, b1, b1, b2, b3, b3, b3, b4, b3, b3, b5]

    max_cue = [False] * num_plots

    config_name_lay = [r'$\beta = 1/4$', r'$\beta = 1/2$', r'$\beta = 1$']
    config_name_3D = [
        r'$\beta = 1/2$', r'$\beta = 1/2$' + ' and ' + r'$\beta = 1$',
        r'$\beta = 1$'
    ]
    config_name_T60 = [
        r'$T_{60} = 0$' + ' s', r'$T_{60} = 0.5$' + ' s',
        r'$T_{60} = 1.0$' + ' s'
    ]

    config_name = [
        config_name_lay, config_name_lay, config_name_lay, config_name_3D,
        config_name_T60, config_name_T60, config_name_T60, config_name_T60,
        config_name_T60, config_name_T60, config_name_T60
    ]

    parameters = {
        'head_rotations': head_rotations,
        'source_exponents': source_exponents,
        'ls_xyz': ls_xyz,
        'directivity_index': directivity_index,
        'room_T60': room_T60,
        'max_cue': max_cue,
        'config_name': config_name
    }

    return parameters
