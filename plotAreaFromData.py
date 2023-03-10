import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from os.path import dirname, join as pjoin
from Utility.ambisonics import sph2cart, cart2sph


def plotAreaFromData(dataset_name, area_len, config_name=[], filetype='pdf'):

    # Define directory for saving figures
    root_dir = dirname(__file__)
    save_path = pjoin(root_dir, 'Figures', 'ListeningArea_IC_ILD')
    fig_data_path = pjoin(root_dir, 'Figures', 'NumpyData')

    DRAW_CENTER_POS_MARKER = True
    DRAW_OFFCENTER_POS_MARKER = True

    # Load data for plotting
    data = np.load(pjoin(fig_data_path, 'Data_' + dataset_name + '.npy'),
                   allow_pickle=True)
    data = data.item()

    LEV_ILD = data['LEV_ILD']
    LEV_IC = data['LEV_IC']

    # Get meshgrid simulation area
    list_X, list_Y = data['list_X'], data['list_Y']
    ls_xyz = data['ls_xyz']
    if config_name != []:
        source_model_names = config_name  # data['source_model_names']
    else:
        source_model_names = data['variable_names']

    ###########################################################

    num_source_models = len(source_model_names)
    scale = 0.8
    fig, axes = plt.subplots(nrows=2,
                             ncols=num_source_models,
                             figsize=(12 * scale, 8 * scale),
                             sharex=True,
                             sharey=False)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    ILD_lvls = np.array([-10, -6, -3, -1, 0])
    IC_lvls = np.array([0.0, 0.2, 0.5, 1.0])

    LEV_ILD = np.clip(LEV_ILD, a_min=-9.9, a_max=0.0)

    font_sz = 18

    # cbar_ticklabels = [['10','6','3','1','0'],  ['1.0', '0.8' , '0.6', '0.4', '0.0'] ]
    cbar_ticklabels = [['10', '6', '3', '1', '0'],
                       ['1.0', '0.8', '0.5', '0.0']]

    cbar_pos_size = [[0.1, 0.54, 0.01, 0.30], [0.1, 0.15, 0.01, 0.30]]
    # cbar_pos_size = [[0.1, 0.65, 0.01, 0.2], [0.1, 0.4, 0.01, 0.2]]

    bmap = colors.ListedColormap(["black", "black", "black", "black"])

    # LOAD FROM NPY Struct
    titles = source_model_names

    vmins = [-10, 0.0, -10]
    vmaxs = [0.0, 1.0, 0.0]

    import matplotlib
    cmap = matplotlib.cm.get_cmap('BuPu_r')

    clrs = [cmap(0.5), cmap(0.6), cmap(0.8), '1.0']

    for r in range(2):
        for c in range(num_source_models):
            if r == 0:
                LEV = LEV_ILD[c, :]
                lvls = ILD_lvls
                axes[r, c].set_title(titles[c], fontsize=font_sz)
                if c == 0:
                    axes[r, c].set_ylabel('|ILD| in dB',
                                          fontsize=font_sz,
                                          labelpad=48)
            if r == 1:
                # Careful: LEV_IC holds 1-IC values for direct plotting
                LEV = LEV_IC[c, :]
                lvls = IC_lvls
                if c == 0:
                    axes[r, c].set_ylabel('IC', fontsize=font_sz, labelpad=48)
            pcm = axes[r, c].pcolormesh(list_X,
                                        list_Y,
                                        LEV,
                                        cmap=cmap,
                                        zorder=1,
                                        vmin=vmins[r],
                                        vmax=vmaxs[r],
                                        shading='gouraud')

            if c == num_source_models - 1:
                color_bar_ax = fig.add_axes(cbar_pos_size[r])
                cbar = fig.colorbar(pcm, cax=color_bar_ax)
                cbar.set_ticks(lvls)
                cbar.set_ticklabels(cbar_ticklabels[r])
                cbar.ax.tick_params(labelsize=12)
                cbar.ax.yaxis.set_ticks_position('left')

            CS = axes[r, c].contour(list_X,
                                    list_Y,
                                    LEV,
                                    levels=lvls,
                                    cmap=bmap,
                                    linewidths=0.5,
                                    zorder=2)
            fmt = {}
            strs = cbar_ticklabels[r]
            for l, s in zip(CS.levels, strs):
                fmt[l] = s

            axes[r, c].yaxis.set_ticks_position("right")
            axes[r, c].clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=12)
            axes[r, c].set_xlim(-(area_len + 0.1), area_len + 0.1)
            axes[r, c].set_ylim(-(area_len + 0.1), area_len + 0.1)
            axes[r,
                 c].set_xticks(np.array([-1.0, -0.5, 0, 0.5, 1.0]) * area_len)
            axes[r, c].set_xticklabels([
                '-' + str(1 * area_len), '-' + str(0.5 * area_len), '0',
                str(0.5 * area_len),
                str(1 * area_len)
            ],
                                       fontsize=12)
            axes[r,
                 c].set_yticks(np.array([-1.0, -0.5, 0, 0.5, 1.0]) * area_len)
            axes[r, c].set_yticklabels([
                '-' + str(1 * area_len), '-' + str(0.5 * area_len), '0',
                str(0.5 * area_len),
                str(1 * area_len)
            ],
                                       fontsize=12)
            if c != 2:
                axes[r, c].set_yticklabels([' '] * 5, fontsize=12)
            if c == 2:
                axes[r, c].set_ylabel('y in m', fontsize=12)
                axes[r, c].yaxis.set_label_position("right")
            if r == 1:
                axes[r, c].set_xlabel('x in m', fontsize=12)

            axes[r, c].grid(visible=True, alpha=0.2)

            # Marker for on-center and off-center position of experiment
            if DRAW_CENTER_POS_MARKER:
                if r == 0:
                    axes[r, c].scatter(0,
                                       0,
                                       s=30,
                                       marker='*',
                                       c='k',
                                       label='{0:.2f}'.format(
                                           round(np.abs(LEV_ILD[c, 22, 22]),
                                                 2)))
                if r == 1:
                    axes[r, c].scatter(0,
                                       0,
                                       s=30,
                                       marker='*',
                                       c='k',
                                       label='{0:.2f}'.format(
                                           round(1 - LEV_IC[c, 22, 22], 2)))
            if DRAW_OFFCENTER_POS_MARKER:
                if r == 0:
                    axes[r, c].scatter(0.5 * area_len,
                                       0,
                                       s=30,
                                       marker='x',
                                       c='k',
                                       label='{0:.2f}'.format(
                                           round(np.abs(LEV_ILD[c, 22, 32]),
                                                 2)))
                if r == 1:
                    axes[r, c].scatter(0.5 * area_len,
                                       0,
                                       s=30,
                                       marker='x',
                                       c='k',
                                       label='{0:.2f}'.format(
                                           round(1 - LEV_IC[c, 22, 32], 2)))
            axes[r, c].legend(loc=(0.755, 0.0),
                              fontsize=12,
                              handletextpad=0.1,
                              handlelength=1.0,
                              labelspacing=0.1,
                              framealpha=0.9,
                              borderpad=0.1)

            # Source / Loudspeaker icon drawing
            t = np.arange(1 / 8, 1, 1 / 4) * 2 * np.pi
            x_square = np.cos(t) * 0.1
            y_square = np.sin(t) * 0.1
            x_tri = np.array([0, 1, -1]) * 0.15
            y_tri = np.array([0, 1, 1]) * 0.15
            square = np.array([x_square, y_square]).transpose()
            tri = np.array([x_tri, y_tri]).transpose()

            sz = 0.5 * area_len
            square = square * sz
            tri = tri * sz

            ls_xyz_norm = ls_xyz / np.tile(
                np.linalg.norm(ls_xyz, axis=1)[:, np.newaxis], (1, 3))
            x = ls_xyz[:, 0]
            y = ls_xyz[:, 1]
            z = ls_xyz[:, 2]
            ls = np.array([x, y]).transpose()
            # tr = ls_coord
            phi, zen = cart2sph(ls_xyz_norm[:, 0], ls_xyz_norm[:, 1],
                                ls_xyz_norm[:, 2])
            alpha = -np.pi / 2 - phi

            rot_mat = np.zeros((phi.shape[0], 2, 2))
            for n in range(phi.shape[0]):
                rot_mat[n, :, :] = np.array(
                    [[np.cos(alpha[n]), -np.sin(alpha[n])],
                     [np.sin(alpha[n]), np.cos(alpha[n])]])

            squares = np.zeros((phi.shape[0], 4, 2))
            tris = np.zeros((phi.shape[0], 3, 2))

            for n in range(phi.shape[0]):
                squares[n, :, :] = np.dot(square, rot_mat[n, :, :])
                tris[n, :, :] = np.dot(tri, rot_mat[n, :, :])

            for n in range(phi.shape[0]):
                axes[r, c].fill(tris[n, :, 0] + ls[n, 0],
                                tris[n, :, 1] + ls[n, 1],
                                color='w',
                                zorder=3,
                                edgecolor='k')
                axes[r, c].fill(squares[n, :, 0] + ls[n, 0],
                                squares[n, :, 1] + ls[n, 1],
                                color='w',
                                zorder=4,
                                edgecolor='k')

    plt.savefig(fname=pjoin(save_path, dataset_name + '.' + filetype),
                bbox_inches='tight',
                dpi=100)
