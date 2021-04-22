# Code created by Claus Naves Eikmeier
# ===========================================================================
# Python Imports
# ===========================================================================
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
# ===========================================================================

# ===========================================================================
# Exportable functions
# ===========================================================================
__all__ = ['grad_plt', 'seismogram_matrix_plt', 'seismogram_plt',
           'seismogram_signals_plt', 'seismograms_freq_plt',
           'signals_plt', 'src_freq_plt', 'src_plt', 'vel_plt',
           'vel_profil_plt', 'xy_multi_plt', 'xy_plt']
# ===========================================================================

# ===========================================================================
# Gradient plot
# ===========================================================================


def grad_plt(grad, shape, spacing, origin, title=None, vmin=None, vmax=None,
             cmap='seismic', figsize=None, dpi=150, save=None, show=True):
    """
    Gradients plot function based on matplotlib.

    Parameters
    ----------
    grad : numpy.ndarray
        Gradients.
    shape : 2-tuple of int
        Model shape. Example: (101, 101).
    spacing : 2-tuple of float
        Model grid spacing (m). Example: (10., 10.).
    origin : 2-tuple of float
        Model origin. Example: (0., 0.).
    title : str, optinal
            Data title. Default is None.
    vmin : float, optional
        Define the data range that the colormap covers.
        Default is None.
    vmax : float, optional
        Define the data range that the colormap covers.
        Default is None.
    cmap : str, optional
        Colormap. Default is 'jet'.
    figsize : 2-tuple, optional
        Figure size. Default is None.
    dpi : int, optional
        Dots per inch in the image. Default is 150.
    save : str, optional
        Save folder path, file name and format. Default is None.
    show : bool, optional
        Image show option. Default is True.

    Returns
    -------
    Without return.
    """

    xi = origin[0]
    xf = (shape[0]-1)*spacing[0]/1000
    yi = origin[1]
    yf = (shape[1]-1)*spacing[1]/1000

    if figsize is not None:
        ax = plt.subplots(figsize=figsize)[1]
    else:
        ax = plt.subplots()[1]

    if title is not None:
        plt.title(title, fontsize='small')

    if vmin is None:
        vmin = -np.amax(np.abs(grad))
    if vmax is None:
        vmax = np.amax(np.abs(grad))

    imag = plt.imshow(np.transpose(grad), vmin=vmin, vmax=vmax,
                      extent=[xi, xf, yf, yi], cmap=cmap)

    ax.set_xlabel('Position (km)')
    ax.set_ylabel('Depth (km)')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(imag, cax=cax, label='Gradient')

    if save is not None:
        plt.savefig(save, dpi=dpi)

    if show is True:
        plt.show()
    else:
        plt.close()
# ===========================================================================

# ===========================================================================
# Seismogram matrix plot in time domain
# ===========================================================================


def seismogram_matrix_plt(seismogram, pseudo_gain=10, win_pick=None,
                          title=None, cmap='gray', figsize=None, dpi=150,
                          save=None, show=True):
    """
    Seismogram matix plot function, based on matplotlib.

    Parameters
    ----------
    seismogram : Devito rec object
        A specific Devito seismogram. Example: seismograms[shot_number].
    pseudo_gain : float, optional
        Image pseudo gain. val = max_value/pseudo_gain, vmin=-val and vmax=val.
        Default is 10.
    win_pick : list, optional
        List of cutoff amplitude index for all receivers.
        The list structure should be [i][j] with:
        i = the receiver number;
        j = the min [0] or max [1] index.
        Default is None.
    title : str, optinal
            Data title. Default is None.
    cmap : str, optional
        Colormap. Default is 'jet'.
    figsize : 2-tuple, optional
        Figure size. Default is None.
    dpi : int, optional
        Dots per inch in the image. Default is 150.
    save : str, optional
        Save folder path, file name and format. Default is None.
    show : bool, optional
        Image show option. Default is True.

    Returns
    -------
    Without return.
    """

    t0 = seismogram.time_range.start
    tn = seismogram.time_range.stop
    dt = seismogram.time_range.step

    xi = 1
    xf = len(seismogram.data[0])

    val = np.amax(np.abs(seismogram.data))/pseudo_gain

    if figsize is not None:
        ax = plt.subplots(figsize=figsize)[1]
    else:
        ax = plt.subplots()[1]

    if title is not None:
        plt.title(title, fontsize='small')

    imag = plt.imshow(seismogram.data, vmin=-val, vmax=val, aspect='auto',
                      extent=[xi, xf, (tn-t0)/1000, 0], cmap=cmap)
    ax.set_xlabel('Trace number')
    ax.set_ylabel('Time (s)')

    # Plot min/max cutoff amplitude
    if win_pick is not None:
        for i in range(len(seismogram.data[0])):
            plt.scatter(i+1, win_pick[i][0]*dt/1000, s=10, c='red', marker='o')
            plt.scatter(i+1, win_pick[i][1]*dt/1000, s=10, c='red', marker='o')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(imag, cax=cax, label='Amplitude')

    if save is not None:
        plt.savefig(save, dpi=dpi)

    if show is True:
        plt.show()
    else:
        plt.close()
# ===========================================================================

# ===========================================================================
# Seismogram plot in time domain
# ===========================================================================


def seismogram_plt(seismogram, gain=1, title=None, figsize=None, dpi=150,
                   save=None, show=True):
    """
    Seismogram plot function based on matplotlib.

    Parameters
    ----------
    seismogram : Devito rec object
        A specific Devito seismogram. Example: seismograms[shot_number].
    gain : float, optional
        Image gain (amplitude*gain). Default is 1.
    title : str, optinal
            Data title. Default is None.
    figsize : 2-tuple, optional
        Figure size. Default is None.
    dpi : int, optional
        Dots per inch in the image. Default is 150.
    save : str, optional
        Save folder path, file name and format. Default is None.
    show : bool, optional
        Image show option. Default is True.

    Returns
    -------
    Without return.
    """

    diff = seismogram.time_range.time_values - seismogram.time_range.start
    time_range_values = diff/1000

    if figsize is not None:
        ax = plt.subplots(figsize=figsize)[1]
    else:
        ax = plt.subplots()[1]

    for i in range(len(seismogram.data[0])):

        x = (i+1)+(seismogram.data[:, i]*gain)

        ax.plot(x, time_range_values, 'k-')
        ax.fill_betweenx(time_range_values, i+1, x, where=(x > (i+1)),
                         color='k')
        ax.fill_betweenx(time_range_values, i+1, x, where=(x < (i+1)),
                         color='r')

    ax.set_xlabel('Trace number')
    ax.set_ylabel('Time (s)')
    ax.grid(axis='y')
    plt.gca().invert_yaxis()
    plt.xlim(0, len(seismogram.data[0])+1)

    if title is not None:
        plt.title(title, fontsize='small')

    if save is not None:
        plt.savefig(save, dpi=dpi)

    if show is True:
        plt.show()
    else:
        plt.close()
# ===========================================================================

# ===========================================================================
# Seimogram plot with several signals
# ===========================================================================


def seismogram_signals_plt(shot_num, seismograms_goal, seismograms_0,
                           rec_num_list=None, seismograms_curr=None,
                           seismograms_i=None, win_pick=None, title=None,
                           xlimit=None, figsize=None, dpi=150, save=None,
                           show=True):
    """
    Seismogram plot function with several signals, based on matplotlib.

    Parameters
    ----------
    shot_num : int
        Shot number. It has to be chosen because it is only possible
        to plot one seismogram.
    seismograms_goal : list
        List of goal seismograms in Devito format.
    seismograms_0 : list
        List of initial seismograms in Devito format.
    rec_num_list : list
        List of receiver numbers. Default is None.
    seismograms_curr : list, optional
        List of current seismograms in Devito format. Default is None.
    seismograms_i : list, optional
        List of intermediate seismograms in Devito format. Default is None.
    win_pick : list, optional
        List of cutoff amplitude index for all shots and receivers.
        The list structure should be [i][j][k] with:
        i = the shot number;
        j = the receiver number;
        k = the min [0] or max [1] index.
        Default is None.
    title : str, optinal
            Data title. Default is None.
    xlimit : 2-tuple, optional
        x axis limit (from, to). Default is None.
    figsize : 2-tuple, optional
        Figure size. Default is None.
    dpi : int, optional
        Dots per inch in the image. Default is 150.
    save : str, optional
        Save folder path, file name and format. Default is None.
    show : bool, optional
        Image show option. Default is True.

    Returns
    -------
    Without return.
    """

    if figsize is None and rec_num_list is None:
        fig, ax = plt.subplots(len(seismograms_goal[shot_num].data[0]), 1,
                               sharex=True)
    elif figsize is not None and rec_num_list is None:
        fig, ax = plt.subplots(len(seismograms_goal[shot_num].data[0]), 1,
                               sharex=True, figsize=figsize)
    elif figsize is None and rec_num_list is not None:
        fig, ax = plt.subplots(len(rec_num_list), 1, sharex=True)
    elif figsize is not None and rec_num_list is not None:
        fig, ax = plt.subplots(len(rec_num_list), 1, sharex=True,
                               figsize=figsize)

    if seismograms_curr is None and seismograms_i is None:
        dt1 = seismograms_goal[0].time_range.step
        dt2 = seismograms_0[0].time_range.step

        x1 = np.arange(
            0, len(seismograms_goal[shot_num].data)*dt1/1000, dt1/1000)
        x2 = np.arange(
            0, len(seismograms_0[shot_num].data)*dt2/1000, dt2/1000)

        if len(x1) > len(seismograms_goal[shot_num].data):
            x1 = np.delete(x1, -1)
        if len(x2) > len(seismograms_0[shot_num].data):
            x2 = np.delete(x2, -1)

        l1 = 0
        l2 = 0

        line_labels = ["Goal signal", "Initial signal"]

        for n, i in enumerate(range(len(seismograms_goal[shot_num].data[0])))\
                if rec_num_list is None else enumerate(rec_num_list):
            signal_goal = seismograms_goal[shot_num].data[:, i]
            signal_0 = seismograms_0[shot_num].data[:, i]

            l1 = ax[n].plot(x1, signal_goal, label='Goal signal',
                            color='green', linestyle='solid',
                            linewidth=0.5)[0]
            l2 = ax[n].plot(x2, signal_0, label='Initial signal',
                            color='red', linestyle='solid',
                            linewidth=0.5)[0]
            ax[n].set_yticks([])
            ax[n].set_ylabel(f'{i+1} ', rotation=0)

            if win_pick is not None:
                ax[n].axvline(win_pick[shot_num][n][0]*dt1/1000, ymin=0.25,
                              ymax=0.75, color='black', lw=1.0, ls='--')
                ax[n].axvline(win_pick[shot_num][n][1]*dt1/1000, ymin=0.25,
                              ymax=0.75, color='black', lw=1.0, ls='--')

    if seismograms_curr is not None and seismograms_i is None:
        dt1 = seismograms_goal[0].time_range.step
        dt2 = seismograms_0[0].time_range.step
        dt3 = seismograms_curr[0].time_range.step

        x1 = np.arange(
            0, len(seismograms_goal[shot_num].data)*dt1/1000, dt1/1000)
        x2 = np.arange(
            0, len(seismograms_0[shot_num].data)*dt2/1000, dt2/1000)
        x3 = np.arange(
            0, len(seismograms_curr[shot_num].data)*dt3/1000, dt3/1000)

        if len(x1) > len(seismograms_goal[shot_num].data):
            x1 = np.delete(x1, -1)
        if len(x2) > len(seismograms_0[shot_num].data):
            x2 = np.delete(x2, -1)
        if len(x3) > len(seismograms_curr[shot_num].data):
            x3 = np.delete(x3, -1)

        l1 = 0
        l2 = 0
        l3 = 0

        line_labels = ["Goal signal", "Initial signal", "Current signal"]

        for n, i in enumerate(range(len(seismograms_goal[shot_num].data[0])))\
                if rec_num_list is None else enumerate(rec_num_list):
            signal_goal = seismograms_goal[shot_num].data[:, i]
            signal_0 = seismograms_0[shot_num].data[:, i]
            signal_curr = seismograms_curr[shot_num].data[:, i]

            l1 = ax[n].plot(x1, signal_goal, label='Goal signal',
                            color='green', linestyle='solid',
                            linewidth=0.5)[0]
            l2 = ax[n].plot(x2, signal_0, label='Initial signal', color='red',
                            linestyle='solid', linewidth=0.5)[0]
            l3 = ax[n].plot(x3, signal_curr, label='Current signal',
                            color='orange', linestyle='solid',
                            linewidth=0.5)[0]
            ax[n].set_yticks([])
            ax[n].set_ylabel(f'{i+1} ', rotation=0)

            if win_pick is not None:
                ax[n].axvline(win_pick[shot_num][n][0]*dt1/1000, ymin=0.25,
                              ymax=0.75, color='black', lw=1.0, ls='--')
                ax[n].axvline(win_pick[shot_num][n][1]*dt1/1000, ymin=0.25,
                              ymax=0.75, color='black', lw=1.0, ls='--')

    if seismograms_curr is None and seismograms_i is not None:
        dt1 = seismograms_goal[0].time_range.step
        dt2 = seismograms_0[0].time_range.step
        dt4 = seismograms_i[0].time_range.step

        x1 = np.arange(
            0, len(seismograms_goal[shot_num].data)*dt1/1000, dt1/1000)
        x2 = np.arange(
            0, len(seismograms_0[shot_num].data)*dt2/1000, dt2/1000)
        x4 = np.arange(
            0, len(seismograms_i[shot_num].data)*dt4/1000, dt4/1000)

        if len(x1) > len(seismograms_goal[shot_num].data):
            x1 = np.delete(x1, -1)
        if len(x2) > len(seismograms_0[shot_num].data):
            x2 = np.delete(x2, -1)
        if len(x4) > len(seismograms_i[shot_num].data):
            x4 = np.delete(x4, -1)

        l1 = 0
        l2 = 0
        l4 = 0

        line_labels = ["Goal signal", "Initial signal", "Intermediate signal"]

        for n, i in enumerate(range(len(seismograms_goal[shot_num].data[0])))\
                if rec_num_list is None else enumerate(rec_num_list):
            signal_goal = seismograms_goal[shot_num].data[:, i]
            signal_0 = seismograms_0[shot_num].data[:, i]
            signal_i = seismograms_i[shot_num].data[:, i]

            l1 = ax[n].plot(x1, signal_goal, label='Goal signal',
                            color='green', linestyle='solid',
                            linewidth=0.5)[0]
            l2 = ax[n].plot(x2, signal_0, label='Initial signal', color='red',
                            linestyle='solid', linewidth=0.5)[0]
            l4 = ax[n].plot(x4, signal_i, label='Intermediate signal',
                            color='black', linestyle='dashed',
                            linewidth=0.5)[0]
            ax[n].set_yticks([])
            ax[n].set_ylabel(f'{i+1} ', rotation=0)

            if win_pick is not None:
                ax[n].axvline(win_pick[shot_num][n][0]*dt1/1000, ymin=0.25,
                              ymax=0.75, color='black', lw=1.0, ls='--')
                ax[n].axvline(win_pick[shot_num][n][1]*dt1/1000, ymin=0.25,
                              ymax=0.75, color='black', lw=1.0, ls='--')

    if seismograms_curr is not None and seismograms_i is not None:
        dt1 = seismograms_goal[0].time_range.step
        dt2 = seismograms_0[0].time_range.step
        dt3 = seismograms_curr[0].time_range.step
        dt4 = seismograms_i[0].time_range.step

        x1 = np.arange(
            0, len(seismograms_goal[shot_num].data)*dt1/1000, dt1/1000)
        x2 = np.arange(
            0, len(seismograms_0[shot_num].data)*dt2/1000, dt2/1000)
        x3 = np.arange(
            0, len(seismograms_curr[shot_num].data)*dt3/1000, dt3/1000)
        x4 = np.arange(
            0, len(seismograms_i[shot_num].data)*dt4/1000, dt4/1000)

        if len(x1) > len(seismograms_goal[shot_num].data):
            x1 = np.delete(x1, -1)
        if len(x2) > len(seismograms_0[shot_num].data):
            x2 = np.delete(x2, -1)
        if len(x3) > len(seismograms_curr[shot_num].data):
            x3 = np.delete(x3, -1)
        if len(x4) > len(seismograms_i[shot_num].data):
            x4 = np.delete(x4, -1)

        l1 = 0
        l2 = 0
        l3 = 0
        l4 = 0

        line_labels = ["Goal signal", "Initial signal", "Current signal",
                       "Intermediate signal"]

        for n, i in enumerate(range(len(seismograms_goal[shot_num].data[0])))\
                if rec_num_list is None else enumerate(rec_num_list):
            signal_goal = seismograms_goal[shot_num].data[:, i]
            signal_0 = seismograms_0[shot_num].data[:, i]
            signal_curr = seismograms_curr[shot_num].data[:, i]
            signal_i = seismograms_i[shot_num].data[:, i]

            l1 = ax[n].plot(x1, signal_goal, label='Goal signal',
                            color='green', linestyle='solid',
                            linewidth=0.5)[0]
            l2 = ax[n].plot(x2, signal_0, label='Initial signal', color='red',
                            linestyle='solid', linewidth=0.5)[0]
            l3 = ax[n].plot(x3, signal_curr, label='Current signal',
                            color='orange', linestyle='solid',
                            linewidth=0.5)[0]
            l4 = ax[n].plot(x4, signal_i, label='Intermediate signal',
                            color='black', linestyle='dashed',
                            linewidth=0.5)[0]
            ax[n].set_yticks([])
            ax[n].set_ylabel(f'{i+1} ', rotation=0)

            if win_pick is not None:
                ax[n].axvline(win_pick[shot_num][n][0]*dt1/1000, ymin=0.25,
                              ymax=0.75, color='black', lw=1.0, ls='--')
                ax[n].axvline(win_pick[shot_num][n][1]*dt1/1000, ymin=0.25,
                              ymax=0.75, color='black', lw=1.0, ls='--')

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False,
                    right=False)

    if title is not None:
        fig.suptitle(title, fontsize='small')

    if xlimit is not None:
        plt.xlim(xlimit[0], xlimit[1])

    plt.xlabel('Time (s)')
    plt.ylabel('Trace number')

    if seismograms_curr is None and seismograms_i is None:
        fig.legend([l1, l2], labels=line_labels, fontsize='medium',
                   loc='upper right', ncol=1)
    if seismograms_curr is not None and seismograms_i is None:
        fig.legend([l1, l2, l3], labels=line_labels, fontsize='medium',
                   loc='upper right', ncol=1)
    if seismograms_curr is None and seismograms_i is not None:
        fig.legend([l1, l2, l4], labels=line_labels, fontsize='medium',
                   loc='upper right', ncol=1)
    if seismograms_curr is not None and seismograms_i is not None:
        fig.legend([l1, l2, l3, l4], labels=line_labels, fontsize='medium',
                   loc='upper right', ncol=1)

    if save is not None:
        plt.savefig(save, dpi=dpi)

    if show is True:
        plt.show()
    else:
        plt.close('all')
# ===========================================================================

# ===========================================================================
# Seismograms plot in frequency domain
# ===========================================================================


def seismograms_freq_plt(seismograms, shot_num=None, title=None, xlimit=None,
                         ylimit=None, figsize=None, dpi=150, save=None,
                         show=True):
    """
    Seismograms in frequency domain plot function, based on matplotlib.

    Parameters
    ----------
    seismograms : list
        List of seismograms in frequency domain (the return of
        seismograms_fft function). The list structur should be:
        [shot number][receiver number]
        [if 0 -> frequencies / if 1 -> amplitudes].
    shot_num : int, optional
        Shot number. If chosen, the related seismogram will be plotted.
        Default is None.
    title : str, optinal
            Data title. Default is None.
    xlimit : 2-tuple, optional
        x axis limit (from, to). Default is None.
    ylimit : 2-tuple, optional
        y axis limit (from, to). Default is None.
    figsize : 2-tuple, optional
        Figure size. Default is None.
    dpi : int, optional
        Dots per inch in the image. Default is 150.
    save : str, optional
        Save folder path, file name and format. Default is None.
    show : bool, optional
        Image show option. Default is True.

    Returns
    -------
    Without return.
    """

    if figsize is not None:
        plt.figure(figsize=figsize)

    if shot_num is None:
        for shot in seismograms:
            for rec in shot:
                plt.plot(rec[0], rec[1], color='k')
    else:
        for rec in seismograms[shot_num]:
            plt.plot(rec[0], rec[1], color='k')

    if xlimit is not None:
        plt.xlim(xlimit[0], xlimit[1])

    if ylimit is not None:
        plt.ylim(ylimit[0], ylimit[1])

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

    if title is not None:
        plt.title(title, fontsize='small')

    if save is not None:
        plt.savefig(save, dpi=dpi)

    if show is True:
        plt.show()
    else:
        plt.close()
# ===========================================================================

# ===========================================================================
# True, initial, guess and interpolated signals plot
# ===========================================================================


def signals_plt(shot_num, rec_num, seismograms_goal, seismograms_0,
                seismograms_curr=None, seismograms_i=None, win_pick=None,
                title=None, xlimit=None, ylimit=None, figsize=None, dpi=150,
                save=None, show=True):
    """
    Seismogram plot function, with several signals. It is based on matplotlib.

    Parameters
    ----------
    shot_num : int
        Shot number. It has to be chosen because it is only possible
        to plot signals from one receiver for a given shot.
    rec_num : int
        Receiver number. It has to be chosen because it is only possible
        to plot signals from one receiver for a given shot.
    seismograms_goal : list
        List of goal seismograms in Devito format.
    seismograms_0 : list
        List of initial seismograms in Devito format.
    seismograms_curr : list, optional
        List of current seismograms in Devito format. Default is None.
    seismograms_i : list, optional
        List of intermediate seismograms in Devito format. Default is None.
    win_pick : list, optional
        List of cutoff amplitude index for all shots and receivers.
        The list structure should be [i][j][k] with:
        i = the shot number;
        j = the receiver number;
        k = the min [0] or max [1] index.
        Default is None.
    title : str, optinal
            Data title. Default is None.
    xlimit : 2-tuple, optional
        x axis limit (from, to). Default is None.
    ylimit : 2-tuple, optional
        y axis limit (from, to). Default is None.
    figsize : 2-tuple, optional
        Figure size. Default is None.
    dpi : int, optional
        Dots per inch in the image. Default is 150.
    save : str, optional
        Save folder path, file name and format. Default is None.
    show : bool, optional
        Image show option. Default is True.

    Returns
    -------
    Without return.
    """

    if figsize is not None:
        plt.figure(figsize=figsize)

    dt_goal = seismograms_goal[shot_num].time_range.step
    dt_0 = seismograms_0[shot_num].time_range.step

    signal_goal = seismograms_goal[shot_num].data[:, rec_num]
    signal_0 = seismograms_0[shot_num].data[:, rec_num]

    x_goal = np.arange(
        0, len(seismograms_goal[shot_num].data)*dt_goal/1000, dt_goal/1000)
    x_0 = np.arange(
        0, len(seismograms_0[shot_num].data)*dt_0/1000, dt_0/1000)

    if len(x_goal) > len(seismograms_goal[shot_num].data):
        x_goal = np.delete(x_goal, -1)
    if len(x_0) > len(seismograms_0[shot_num].data):
        x_0 = np.delete(x_0, -1)

    plt.plot(x_goal, signal_goal, label='Goal signal', color='green',
             linestyle='solid', linewidth=1.0)
    plt.plot(x_0, signal_0, label='Initial signal', color='red',
             linestyle='solid', linewidth=1.0)

    if seismograms_i is not None:
        dt_i = seismograms_i[shot_num].time_range.step
        signal_i = seismograms_i[shot_num].data[:, rec_num]
        x_i = np.arange(
            0, len(seismograms_i[shot_num].data)*dt_i/1000, dt_i/1000)
        if len(x_i) > len(seismograms_i[shot_num].data):
            x_i = np.delete(x_i, -1)
        plt.plot(x_i, signal_i, label='Intermediate signal', color='black',
                 linestyle='dashed', linewidth=1.0)

    if seismograms_curr is not None:
        dt_curr = seismograms_curr[shot_num].time_range.step
        signal_curr = seismograms_curr[shot_num].data[:, rec_num]
        x_curr = np.arange(
            0, len(seismograms_curr[shot_num].data)*dt_curr/1000, dt_curr/1000)
        if len(x_curr) > len(seismograms_curr[shot_num].data):
            x_curr = np.delete(x_curr, -1)
        plt.plot(x_curr, signal_curr, label='Current signal', color='orange',
                 linestyle='solid', linewidth=1.0)

    if win_pick is not None:
        plt.axvline(win_pick[shot_num][rec_num][0]*dt_goal/1000, ymin=0.25,
                    ymax=0.75, color='black', lw=1.5, ls='--')
        plt.axvline(win_pick[shot_num][rec_num][1]*dt_goal/1000, ymin=0.25,
                    ymax=0.75, color='black', lw=1.5, ls='--')

    if title is not None:
        plt.title(title, fontsize='small')

    if xlimit is not None:
        plt.xlim(xlimit[0], xlimit[1])

    if ylimit is not None:
        plt.ylim(ylimit[0], ylimit[1])

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    if save is not None:
        plt.savefig(save, dpi=dpi)

    if show is True:
        plt.show()
    else:
        plt.close('all')
# ===========================================================================

# ===========================================================================
# Source plot in frequency domain
# ===========================================================================


def src_freq_plt(src, title=None, xlimit=None, ylimit=None, color='k',
                 figsize=None, dpi=150, save=None, show=True):
    """
    Source in frequency domain plot function, based on matplotlib.

    Parameters
    ----------
    src : list
        List with source in frequency domain (the return of
        src_fft function). The list structur should be:
        [if 0 -> frequencies / if 1 -> amplitudes].
    title : str, optinal
            Data title. Default is None.
    xlimit : 2-tuple, optional
        x axis limit (from, to). Default is None.
    ylimit : 2-tuple, optional
        y axis limit (from, to). Default is None.
    color : str, optional
        Graphic color. Default is 'k' (black).
    figsize : 2-tuple, optional
        Figure size. Default is None.
    dpi : int, optional
        Dots per inch in the image. Default is 150.
    save : str, optional
        Save folder path, file name and format. Default is None.
    show : bool, optional
        Image show option. Default is True.

    Returns
    -------
    Without return.
    """

    if figsize is not None:
        plt.figure(figsize=figsize)

    plt.plot(src[0], src[1], color='k')

    if title is not None:
        plt.title(title, fontsize='small')

    if xlimit is not None:
        plt.xlim(xlimit[0], xlimit[1])

    if ylimit is not None:
        plt.ylim(ylimit[0], ylimit[1])

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

    if save is not None:
        plt.savefig(save, dpi=dpi)

    if show is True:
        plt.show()
    else:
        plt.close()
# ===========================================================================

# ===========================================================================
# Source plot in time domain
# ===========================================================================


def src_plt(src, title=None, xlimit=None, ylimit=None, color='k', figsize=None,
            dpi=150, save=None, show=True):
    """
    Source plot function based on matplotlib.

    Parameters
    ----------
    src : Source
        Object with the source information.
    title : str, optinal
            Data title. Default is None.
    xlimit : 2-tuple, optional
        x axis limit (from, to). Default is None.
    ylimit : 2-tuple, optional
        y axis limit (from, to). Default is None.
    color : str, optional
        Graphic color. Default is 'k' (black).
    figsize : 2-tuple
        Figure size. Default is None.
    dpi : int, optional
        Dots per inch in the image. Default is 150.
    save : str, optional
        Save folder path, file name and format. Default is None.
    show : bool, optional
        Image show option. Default is True.

    Returns
    -------
    Without return.
    """

    if figsize is not None:
        plt.figure(figsize=figsize)

    time_range_values = src.time_range.time_values - src.time_range.start
    plt.plot(time_range_values, src.data[:, 0], color=color)

    if title is not None:
        plt.title(title, fontsize='small')

    if xlimit is not None:
        plt.xlim(xlimit[0], xlimit[1])

    if ylimit is not None:
        plt.ylim(ylimit[0], ylimit[1])

    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')

    if save is not None:
        plt.savefig(save, dpi=dpi)

    if show is True:
        plt.show()
    else:
        plt.close()
# ===========================================================================

# ===========================================================================
# Velocity plot
# ===========================================================================


def vel_plt(vel, shape, spacing, origin, shots=None, receivers=None,
            title=None, vmin=None, vmax=None, cmap='jet', figsize=None,
            dpi=150, save=None, show=True):
    """
    Velocity model plot function based on matplotlib.

    Parameters
    ----------
    vel : numpy.ndarray
        Velocity model.
    shape : 2-tuple of int
        Model shape. Example: (101, 101).
    spacing : 2-tuple of float
        Model grid spacing (m). Example: (10., 10.).
    origin : 2-tuple of float
        Model origin. Example: (0., 0.).
    shots : numpy.ndarray, optional
        Shots position. Default is None.
    receivers : numpy.ndarray, optional
        Receivers position. Default is None.
    title : str, optinal
            Data title. Default is None.
    vmin : float, optional
        Define the data range that the colormap covers.
        Default is None.
    vmax : float, optional
        Define the data range that the colormap covers.
        Default is None.
    cmap : str, optional
        Colormap. Default is 'jet'.
    figsize : 2-tuple, optional
        Figure size. Default is None.
    dpi : int, optional
        Dots per inch in the image. Default is 150.
    save : str, optional
        Save folder path, file name and format. Default is None.
    show : bool, optional
        Image show option. Default is True.

    Returns
    -------
    Without return.
    """

    xi = origin[0]
    xf = (shape[0]-1)*spacing[0]/1000
    yi = origin[1]
    yf = (shape[1]-1)*spacing[1]/1000

    if figsize is not None:
        ax = plt.subplots(figsize=figsize)[1]
    else:
        ax = plt.subplots()[1]

    if title is not None:
        plt.title(title, fontsize='small')

    imag = plt.imshow(np.transpose(vel), extent=[xi, xf, yf, yi], cmap=cmap,
                      vmin=vmin, vmax=vmax)
    ax.set_xlabel('Position (km)')
    ax.set_ylabel('Depth (km)')

    # Plot receiver points, if provided
    if receivers is not None:
        plt.scatter(1e-3*receivers[:, 0], 1e-3*receivers[:, 1],
                    s=25, c='green', marker='v')

    # Plot shots points, if provided
    if shots is not None:
        plt.scatter(1e-3*shots[:, 0], 1e-3*shots[:, 1],
                    s=25, c='red', marker='*')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(imag, cax=cax, label='Velocity (km/s)')

    if save is not None:
        plt.savefig(save, dpi=dpi)

    if show is True:
        plt.show()
    else:
        plt.close()
# ===========================================================================

# ===========================================================================
# Velocity profile plot
# ===========================================================================


def vel_profil_plt(vel, shape, spacing, origin, position, title=None,
                   label=None, xlimit=None, ylimit=None, linewidth=0.5,
                   figsize=None, dpi=150, save=None, show=True):
    """
    Velocity profile for a list of velocity models, based on matplotlib.

    Parameters
    ----------
    vel : list of numpy.ndarray
        List of velocity model.
    shape : 2-tuple of int
        Model shape. Example: (101, 101).
    spacing : 2-tuple of float
        Model grid spacing (m). Example: (10., 10.).
    origin : 2-tuple of float
        Model origin. Example: (0., 0.).
    position : float
        Profile position (m).
    title : str, optinal
            Data title. Default is None.
    label : list of str
        Data label list. Default is None.
    xlimit : 2-tuple, optional
        x axis limit (from, to). Default is None.
    ylimit : 2-tuple, optional
        y axis limit (from, to). Default is None.
    figsize : 2-tuple
        Figure size. Default is None.
    dpi : int, optional
        Dots per inch in the image. Default is 150.
    save : str, optional
        Save folder path, file name and format. Default is None.
    show : bool, optional
        Image show option. Default is True.

    Returns
    -------
    Without return.
    """

    y = np.linspace(
        origin[1], (shape[1]-1)*spacing[1]/1000 + origin[1], num=shape[1])

    x_pos = round(position/spacing[1])

    if figsize is not None:
        plt.figure(figsize=figsize)

    if label is None:
        for i in range(len(vel)):
            plt.plot(vel[i][x_pos], y, linewidth=linewidth)
    else:
        for i in range(len(vel)):
            plt.plot(vel[i][x_pos], y, label=label[i], linewidth=linewidth)

    if title is not None:
        plt.title(title, fontsize='small')

    if xlimit is not None:
        plt.xlim(xlimit[0], xlimit[1])

    if ylimit is not None:
        plt.ylim(ylimit[0], ylimit[1])

    plt.xlabel('Velocity (km/s)')
    plt.ylabel('Depth (km)')
    plt.gca().invert_yaxis()

    if label is not None:
        if len(label) <= 5:
            plt.legend(fontsize="medium")
        if len(label) > 5 and len(label) <= 10:
            plt.legend(fontsize="small", ncol=2)
        if len(label) > 10:
            plt.legend(fontsize="x-small", ncol=2)

    if save is not None:
        plt.savefig(save, dpi=dpi)

    if show is True:
        plt.show()
    else:
        plt.close()
# ===========================================================================

# ===========================================================================
# Standard multidata y versus x plot
# ===========================================================================


def xy_multi_plt(xy, title=None, label=None, xlabel=None, ylabel=None,
                 xlimit=None, ylimit=None, figsize=None, dpi=150, save=None,
                 show=True):
    """
    Standard y versus x plot for a list of data, based on matplotlib.

    Parameters
    ----------
    xy : list of numpy.ndarray
        Data list.
    title : str, optinal
            Data title. Default is None.
    label : list of str
        Data label list. Default is None.
    xlabel : str
        x axis label. Default is None.
    ylabel : str
        y axis label. Default is None.
    xlimit : 2-tuple, optional
        x axis limit (from, to). Default is None.
    ylimit : 2-tuple, optional
        y axis limit (from, to). Default is None.
    figsize : 2-tuple
        Figure size. Default is None.
    dpi : int, optional
        Dots per inch in the image. Default is 150.
    save : str, optional
        Save folder path, file name and format. Default is None.
    show : bool, optional
        Image show option. Default is True.

    Returns
    -------
    Without return.
    """

    if figsize is not None:
        plt.figure(figsize=figsize)

    if label is None:
        for i in range(len(xy)):
            plt.plot(xy[i])
    else:
        for i in range(len(xy)):
            plt.plot(xy[i], label=label[i])

    if title is not None:
        plt.title(title, fontsize='small')

    if xlimit is not None:
        plt.xlim(xlimit[0], xlimit[1])

    if ylimit is not None:
        plt.ylim(ylimit[0], ylimit[1])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if label is not None:
        if len(label) <= 5:
            plt.legend(fontsize="medium")
        if len(label) > 5 and len(label) <= 10:
            plt.legend(fontsize="small", ncol=2)
        if len(label) > 10:
            plt.legend(fontsize="x-small", ncol=2)

    if save is not None:
        plt.savefig(save, dpi=dpi)

    if show is True:
        plt.show()
    else:
        plt.close()
# ===========================================================================

# ===========================================================================
# Standard y versus x plot
# ===========================================================================


def xy_plt(xy, title=None, xlabel=None, ylabel=None, xlimit=None, ylimit=None,
           color='k', figsize=None, dpi=150, save=None, show=True):
    """
    Standard y versus x plot based on matplotlib.

    Parameters
    ----------
    xy : numpy.ndarray
        Data.
    title : str, optinal
            Data title. Default is None.
    xlabel : str
        x axis label. Default is None.
    ylabel : str
        y axis label. Default is None.
    xlimit : 2-tuple, optional
        x axis limit (from, to). Default is None.
    ylimit : 2-tuple, optional
        y axis limit (from, to). Default is None.
    color : str, optional
        Graphic color. Default is 'k' (black).
    figsize : 2-tuple
        Figure size. Default is None.
    dpi : int, optional
        Dots per inch in the image. Default is 150.
    save : str, optional
        Save folder path, file name and format. Default is None.
    show : bool, optional
        Image show option. Default is True.

    Returns
    -------
    Without return.
    """

    if figsize is not None:
        plt.figure(figsize=figsize)

    plt.plot(xy, color=color)

    if title is not None:
        plt.title(title, fontsize='small')

    if xlimit is not None:
        plt.xlim(xlimit[0], xlimit[1])

    if ylimit is not None:
        plt.ylim(ylimit[0], ylimit[1])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if save is not None:
        plt.savefig(save, dpi=dpi)

    if show is True:
        plt.show()
    else:
        plt.close()
# ===========================================================================
