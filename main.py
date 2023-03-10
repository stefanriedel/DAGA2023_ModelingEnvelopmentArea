from paper_parameter_list import generateParameterList
import numpy as np
from os.path import dirname, join as pjoin
import os

from computeAndSaveCues import computeAndSaveCues
from plotAreaFromData import plotAreaFromData

root_dir = dirname(__file__)
save_path = pjoin(root_dir, 'Figures', 'ListeningArea_IC_ILD')
fig_data_path = pjoin(root_dir, 'Figures', 'NumpyData')
utility_path = pjoin(root_dir, 'Utility')
hrir_path = os.path.join(root_dir, 'Utility', 'IRs')

N_ambi = 5
room_volume = 5 * 10 * 11
area_len = 5

# Perceptually meaningful frequency ranges to average IC and ILD,
# where auditory perception is most sensitive to ITD and ILD cues.
# Refer to DAGA paper for references/discussion.
IC_range = np.array([200, 1600])
ILD_range = np.array([200, 12800])

# Parameter list used for plots in the paper
parameters = generateParameterList()

for idx in range(6):  # render the plots of the paper
    ls_xyz = parameters['ls_xyz'][idx]
    source_exponents = parameters['source_exponents'][idx]
    config_name = parameters['config_name'][idx]
    head_rotations = parameters['head_rotations'][idx]
    room_T60 = parameters['room_T60'][idx]
    directivity_index = parameters['directivity_index'][idx]
    max_cue = parameters['max_cue'][idx]

    dataset_name = str(idx + 1) + '_Nambi' + str(N_ambi) + 'Head0'

    COMPUTE = True
    PLOT = True

    if COMPUTE:
        computeAndSaveCues(ls_xyz, source_exponents, config_name,
                           head_rotations, room_volume, room_T60,
                           directivity_index, IC_range, ILD_range, area_len,
                           fig_data_path, hrir_path, N_ambi, max_cue,
                           dataset_name)
    if PLOT:
        plotAreaFromData(dataset_name, area_len, config_name, 'png')
