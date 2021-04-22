# Code created by Claus Naves Eikmeier

# ===========================================================================
# Python Imports
# ===========================================================================
import numpy as np
from scipy import interpolate, signal, stats
import copy
# ===========================================================================

# ===========================================================================
# Devito Imports
# ===========================================================================
from examples.seismic import PointSource, Model
# ===========================================================================

# ===========================================================================
# Exportable functions
# ===========================================================================
__all__ = ['critical_dt', 'frequency_band_info', 'reshape_shape',
           'reshape_spacing', 'src_freq_filt', 'seismograms_fft',
           'sg_freq_filt', 'sgs_freq_filt', 'sgs_freq_normalization',
           'sgs_normalization', 'spacing_info', 'src_fft']
# ===========================================================================

# ===========================================================================
# Critical dt calculation function
# ===========================================================================


def critical_dt(model_orig, model_t, setup):
    """
    Critical time step calculation based on the Nyquist frequency, original
    and true model.

    Parameters
    ----------
    model_orig : Model
        Object with the physical parameters of the original model.
    model_t : Model
        Object with the physical parameters of the true model.
    setup : Setup
        Object with setup informations.

    Returns
    -------
    Critical time step in ms.
    """

    # Nyquist dt
    dt_ny = (1/(2*setup.f_nyquist*1000))*1000  # ms

    # Temporary model for calculating the critical dt based on the minimum
    # and maximum velocity considered
    vp_temp = np.empty(setup.shape)
    middle_temp = int(setup.shape[1]/2)
    vp_temp[:, :middle_temp] = setup.vmin
    vp_temp[:, middle_temp:] = setup.vmax

    model_temp = Model(vp=vp_temp, origin=setup.origin, shape=setup.shape,
                       spacing=setup.spacing, space_order=setup.space_order,
                       nbl=setup.nbl, bcs=setup.bcs, grid=setup.grid)

    dt_list = [model_temp.critical_dt, model_orig.critical_dt,
               model_t.critical_dt, dt_ny]

    return np.amin(dt_list)
# ===========================================================================

# ===========================================================================
# Signal frequency band information function
# ===========================================================================


def frequency_band_info(seismograms, factor, show=True):
    """
    First, for each seismogram, the minimum frequency for amplitude*factor,
    the maximum frequency for amplitude*factor and the peak frequency
    are calculated. Subsequently, an average of the minimum, maximum
    and peak frequencies is obtained.

    Parameters
    ----------
    seismograms : Seismograms
        Object with the seismograms.
    factor : float
        Amplitude factor. For example, if 10% of the maximum
        amplitude is desired, the factor needs to be 0.1.
    show : bool, optional
        Data print option. Default is True.

    Returns
    -------
    1) Average of the minimum frequencies.
    2) Average of the maximum frequencies.
    3) Average of the peak frequencies.
    """

    freq_min = []
    freq_max = []
    freq_peak = []

    # Obtaining seismograms in the frequency domain
    seismograms_freq = seismograms_fft(seismograms)

    nshots = len(seismograms)
    nreceivers = len(seismograms[0].data[0])

    for i in range(nshots):

        for j in range(nreceivers):

            # Calculation of peak frequencies for maximum amplitudes and
            # cut-off frequencies based on a percentage of the maximum
            # amplitudes
            amp_max = np.amax(seismograms_freq[i][j][1])
            cut_amp = amp_max*factor

            idx = np.where(seismograms_freq[i][j][1] == amp_max)
            freq_peak.append(seismograms_freq[i][j][0][idx])

            # Here the seismogram is scanned from the lowest frequency forward
            # until the cut-off frequency is found (that is, the minimum
            # cut-off frequency)
            for k in range(len(seismograms_freq[i][j][1])-1):
                if seismograms_freq[i][j][1][k] >= cut_amp:
                    freq_min.append(seismograms_freq[i][j][0][k])
                    break

            # Here the seismogram is scanned from the highest frequency
            # backward until the cut-off frequency is found (that is, the
            # maximum cut-off frequency)
            for m in range(len(seismograms_freq[i][j][1])-1, 0, -1):
                if seismograms_freq[i][j][1][m] >= cut_amp:
                    freq_max.append(seismograms_freq[i][j][0][m])
                    break

    if show is True:
        print(f'Minimum frequency average = {np.average(freq_min)}Hz')
        print(f'Maximum frequency average = {np.average(freq_max)}Hz')
        print(f'Peak frequency average = {np.average(freq_peak)}Hz')

    # An average of the frequencies is returned
    return np.average(freq_min), np.average(freq_max), np.average(freq_peak)
# ===========================================================================

# ===========================================================================
# Velocity model reshape function, with shape information
# ===========================================================================


def reshape_shape(v, new_shape):
    """
    Velocity model reshape function, based on SciPy
    scipy.interpolate.interp1d.

    Parameters
    ----------
    v : numpy.ndarray
        Velocity model.
    new_shape : 2-tuple of int
        New shape for the velocity model.

    Returns
    -------
    Velocity model with the new shape.
    """

    x1 = np.array(range(v.shape[0]))
    x2 = np.linspace(x1.min(), x1.max(), new_shape[0])
    f1 = interpolate.interp1d(x1, v, axis=0, kind='nearest')
    v_aux = (f1(x2))

    y1 = np.array(range(v_aux.shape[1]))
    y2 = np.linspace(y1.min(), y1.max(), new_shape[1])
    f2 = interpolate.interp1d(y1, v_aux, axis=1, kind='nearest')
    v_out = (f2(y2))

    return v_out
# ===========================================================================

# ===========================================================================
# Velocity model reshape function, with spacing information
# ===========================================================================


def reshape_spacing(v, old_spacing, new_spacing):
    """
    Velocity model reshape function, based on SciPy
    scipy.interpolate.interp1d.

    Parameters
    ----------
    v : numpy.ndarray
        Velocity model.
    old_spacing : 2-tuple of int
        Old spacing of the velocity model.
    new_spacing : 2-tuple of int
        New spacing of the velocity model.

    Returns
    -------
    Velocity model with the new shape.
    """

    new_shape = [None, None]
    new_shape[0] = int(v.shape[0]*old_spacing[0]/new_spacing[0]) + 1
    new_shape[1] = int(v.shape[1]*old_spacing[1]/new_spacing[1]) + 1

    x1 = np.array(range(v.shape[0]))
    x2 = np.linspace(x1.min(), x1.max(), new_shape[0])
    f1 = interpolate.interp1d(x1, v, axis=0, kind='nearest')
    v_aux = (f1(x2))

    y1 = np.array(range(v_aux.shape[1]))
    y2 = np.linspace(y1.min(), y1.max(), new_shape[1])
    f2 = interpolate.interp1d(y1, v_aux, axis=1, kind='nearest')
    v_out = (f2(y2))

    return v_out
# ===========================================================================

# ===========================================================================
# Seismograms FFT function
# ===========================================================================


def seismograms_fft(seismograms):
    """
    Fast Fourier transform in time domain seismograms.

    Parameters
    ----------
    seismograms : Seismograms
        Object with the seismograms.

    Returns
    -------
    Seismograms in frequency domain. The seimograms list structur is:
    [shot number][receiver number][if 0 -> frequencies / if 1 -> amplitudes].
    """

    seismograms_freq = []

    nshots = len(seismograms)
    nreceivers = len(seismograms[0].data[0])

    for i in range(nshots):

        seismograms_freq.append([])

        for j in range(nreceivers):

            seismograms_freq[i].append([])

            amp_time = seismograms[i].data[:, j]
            amp_freq = np.fft.rfft(amp_time)
            freq = np.fft.rfftfreq(
                len(amp_time), d=seismograms[0].time_range.step*0.001)
            seismograms_freq[i][j].append(freq)
            seismograms_freq[i][j].append(np.abs(amp_freq))

    return seismograms_freq
# ===========================================================================

# ===========================================================================
# Seismogram frquency filter
# ===========================================================================


def sg_freq_filt(seismogram, cutoff_freq, filt_type='lowpass', filt_order=2):
    """
    Lowpass frequency filter for a seismogram.

    Parameters
    ----------
    seismogram : Seismogram
        Object with the seismogram.
    cutoff_freq : float or 2-tuple
        Cutoff frequency or frequencies. For lowpass and highpass filters,
        cutoff_freq is a float. For bandpass and bandstop filters,
        cutoff_freq is a 2-tuple.
    filt_type : str, optional
        The type of the filter (‘lowpass’, ‘highpass’, ‘bandpass’
        or ‘bandstop’). Default is 'lowpass'.
    filt_order : int, optional
        The order of the filter. Default is 2.

    Returns
    -------
    Filtered seismogram.
    """

    seismogram_filtered = copy.deepcopy(seismogram)

    if cutoff_freq is not None:
        sos = signal.butter(filt_order, cutoff_freq, btype=filt_type,
                            fs=1/(seismogram.time_range.step*0.001),
                            output='sos')

        for j in range(len(seismogram_filtered.data[0])):
            seismogram_filtered.data[:, j] = signal.sosfiltfilt(
                sos, seismogram.data[:, j])

    return seismogram_filtered
# ===========================================================================

# ===========================================================================
# Seismograms frquency filter
# ===========================================================================


def sgs_freq_filt(seismograms, cutoff_freq, filt_type='lowpass', filt_order=2):
    """
    Lowpass frequency filter for seismograms.

    Parameters
    ----------
    seismograms : Seismograms
        Object with the seismograms.
    cutoff_freq : float or length-2 tuple
        Cutoff frequency or frequencies. For lowpass and highpass filters,
        cutoff_freq is a float. For bandpass and bandstop filters,
        cutoff_freq is a length-2 tuple.
    filt_type : str, optional
        The type of the filter (‘lowpass’, ‘highpass’, ‘bandpass’
        or ‘bandstop’). Default is 'lowpass'.
    filt_order : int, optional
        The order of the filter. Default is 2.

    Returns
    -------
    Filtered seismograms.
    """

    seismograms_filtered = copy.deepcopy(seismograms)

    if cutoff_freq is not None:
        sos = signal.butter(filt_order, cutoff_freq, btype=filt_type,
                            fs=1/(seismograms[0].time_range.step*0.001),
                            output='sos')

        for i in range(len(seismograms)):
            for j in range(len(seismograms[0].data[0])):
                seismograms_filtered[i].data[:, j] = signal.sosfiltfilt(
                    sos, seismograms[i].data[:, j])

    return seismograms_filtered
# ===========================================================================

# ===========================================================================
# Frequency domain seismograms normalization function
# ===========================================================================


def sgs_freq_normalization(seismograms_freq, norm_type='z_score'):
    """
    Frequency domain seismograms normalization function.

    Parameters
    ----------
    seismograms : Seismograms
        Seismograms in frequency domain (return of seismograms_fft()).
    norm_type : str, optional
        Normalization type. The options are: 'min_max' or 'z_score'.
        Default is 'z_score'.

    Returns
    -------
    Normalized frequency domain seismograms. The seimograms list structur is:
    [shot number][receiver number][if 0 -> frequencies / if 1 -> amplitudes].
    """

    sgsf = copy.deepcopy(seismograms_freq)

    nshots = len(sgsf)
    nreceivers = len(sgsf[0])
    npoints = len(sgsf[0][0][1])

    if norm_type == 'min_max':
        for i in range(nshots):
            for j in range(nreceivers):
                min = np.amin(sgsf[i][j][1])
                max = np.amax(sgsf[i][j][1])
                for k in range(npoints):
                    sgsf[i][j][1][k] = \
                        (sgsf[i][j][1][k]-min)/(max-min)

    if norm_type == 'z_score':
        for i in range(nshots):
            for j in range(nreceivers):
                sgsf[i][j][1] = stats.zscore(
                    sgsf[i][j][1])

    return sgsf
# ===========================================================================

# ===========================================================================
# Time domain seismograms normalization function
# ===========================================================================


def sgs_normalization(seismograms, norm_type='z_score'):
    """
    Devito seismograms signals normalization function.

    Parameters
    ----------
    seismograms : Seismograms
        Object with the seismograms.
    norm_type : str, optional
        Normalization type. The options are: 'min_max' or 'z_score'.
        Default is 'z_score'.

    Returns
    -------
    Normalized seismograms.
    """

    sgs = copy.deepcopy(seismograms)

    nshots = len(sgs)
    nreceivers = len(sgs[0].data[0])
    npoints = len(sgs[0].data)

    if norm_type == 'min_max':
        for i in range(nshots):
            for j in range(nreceivers):
                min = np.amin(sgs[i].data[:, j])
                max = np.amax(sgs[i].data[:, j])
                for k in range(npoints):
                    sgs[i].data[k][j] = (
                        sgs[i].data[k][j]-min)/(max-min)

    if norm_type == 'z_score':
        for i in range(nshots):
            for j in range(nreceivers):
                sgs[i].data[:, j] = stats.zscore(
                    sgs[i].data[:, j])

    return sgs
# ===========================================================================

# ===========================================================================
# Model grid spacing information
# ===========================================================================


def spacing_info(freq_peak, freq_max, setup):
    """
    Function for verification if the model grid spacing is adequate.

    Parameters
    ----------
    freq_peak : float
        Peak frequency.
    freq_max : float
        Maximum frequency.
    setup : Setup
        Object with setup informations.

    Returns
    -------
    Whitout return.
    """

    lam_peak = setup.vmin*1000/freq_peak
    lam_min = setup.vmin*1000/freq_max

    spacing_max_2ord = lam_peak/20
    spacing_max_4ord = lam_peak/12
    critical_spacing = lam_min/2

    if setup.space_order == 2:
        if (setup.spacing[0] <= spacing_max_2ord and
                setup.spacing[1] <= spacing_max_2ord):
            if (setup.spacing[0] > critical_spacing or
                    setup.spacing[1] > critical_spacing):
                print(f'INADEQUATE spacing. dx={setup.spacing[0]}' +
                      f' | dz={setup.spacing[1]} | ' +
                      f'critical_spacing={critical_spacing}')
            else:
                print(f'ADEQUATE spacing. dx={setup.spacing[0]}' +
                      f' | dz={setup.spacing[1]} | ' +
                      f'spacing_max_2ord={spacing_max_2ord}')
        else:
            print(f'INADEQUATE spacing. dx={setup.spacing[0]}' +
                  f' | dz={setup.spacing[1]} | ' +
                  f'spacing_max_2ord={spacing_max_2ord}')

    if setup.space_order == 4:
        if (setup.spacing[0] <= spacing_max_4ord and
                setup.spacing[1] <= spacing_max_4ord):
            if (setup.spacing[0] > critical_spacing or
                    setup.spacing[1] > critical_spacing):
                print(f'INADEQUATE spacing. dx={setup.spacing[0]}' +
                      f' | dz={setup.spacing[1]} | ' +
                      f'critical_spacing={critical_spacing}')
            else:
                print(f'ADEQUATE spacing. dx={setup.spacing[0]}' +
                      f' | dz={setup.spacing[1]} | ' +
                      f'spacing_max_4ord={spacing_max_4ord}')
        else:
            print(f'INADEQUATE spacing. dx={setup.spacing[0]}' +
                  f' | dz={setup.spacing[1]} | ' +
                  f'spacing_max_4ord={spacing_max_4ord}')

    if setup.space_order != 2 and setup.space_order != 4:
        if (setup.spacing[0] > critical_spacing or
                setup.spacing[1] > critical_spacing):
            print(f'INADEQUATE spacing. dx={setup.spacing[0]}' +
                  f' | dz={setup.spacing[1]} | ' +
                  f'critical_spacing={critical_spacing}')
        else:
            print('Unable to inform appropriate spacing information ' +
                  'for the current space order.')
# ===========================================================================

# ===========================================================================
# Source FFT function
# ===========================================================================


def src_fft(src):
    """
    Fast Fourier transform in time domain source.

    Parameters
    ----------
    src : Source
        Object with the source information.

    Returns
    -------
    Source in frequency domain. The source list structur is:
    [if 0 -> frequencies / if 1 -> amplitudes].
    """

    src_freq = []

    amp_time = src.data[:, 0]
    amp_freq = np.fft.rfft(amp_time)
    freq = np.fft.rfftfreq(len(amp_time), d=src.time_range.step*0.001)
    src_freq.append(freq)
    src_freq.append(np.abs(amp_freq))

    return src_freq
# ===========================================================================

# ===========================================================================
# Source frquency filter
# ===========================================================================


def src_freq_filt(src, cutoff_freq, filt_type='lowpass', filt_order=2):
    """
    Lowpass frequency filter function for the source.

    Parameters
    ----------
    src : Source
        Object with the source information.
    cutoff_freq : float or 2-tuple
        Cutoff frequency or frequencies. For lowpass and highpass filters,
        cutoff_freq is a float. For bandpass and bandstop filters,
        cutoff_freq is a 2-tuple.
    filt_type : str, optional
        The type of the filter (‘lowpass’, ‘highpass’, ‘bandpass’
        or ‘bandstop’). Default is 'lowpass'.
    filt_order : int, optional
        The order of the filter. Default is 2.

    Returns
    -------
    Filtered source.
    """

    src_filtered = PointSource(name='src', grid=src.grid, npoint=1,
                               time_range=src.time_range)

    sos = signal.butter(filt_order, cutoff_freq, btype=filt_type,
                        fs=1/(src.time_range.step*0.001), output='sos')
    src_filtered.data[:, 0] = signal.sosfiltfilt(sos, src.data[:, 0])

    return src_filtered
# ===========================================================================
