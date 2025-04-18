#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=C0103,C0301
"""
Copyright 2020-2023 Ivo Alxneit (ivo.alxneit@psi.ch)

This file is part of JEOL_eds.
JEOL_eds is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

JEOL_eds is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with JEOL_eds. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import numpy as np
from scipy import ndimage
from skimage.measure import profile_line
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgba
from matplotlib.ticker import AutoMinorLocator
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def filter_isolated_pixels(array, struct=np.ones((3, 3))):
    """ Return array with completely isolated single cells removed.

    Parameters
    ----------
    array : Ndarray
        Array with completely isolated single cells
    struct : Tuple(tuple)
        Structure array for generating unique regions. Default region is 3x3,
        i.e. only the direct neighbors matter.

    Returns
    -------
    filtered_array : Ndarray
        Input array with isolated cells removed.

    Notes
    -----
        Copied almost verbatim from
        https://stackoverflow.com/questions/28274091/removing-completely-isolated-cells-from-python-array

    Examples
    --------

    >>> from JEOL_eds import JEOL_pts
    >>> import JEOL_eds.utils as JU

    Load data
    >>> dc = JEOL_pts('data/64.pts', split_frames=True)

    Si map of first frame
    >>> m = dc.map(interval=(1.50, 1.80), energy=True, frames=[0])

    How many pixels with no direct neighbors (most probably noise) are present?
    >>> (m - JU.filter_isolated_pixels(m)).sum()
    np.float64(93.0)
    """
    filtered_array = np.copy(array)
    id_regions, num_ids = ndimage.label(filtered_array, structure=struct)
    id_sizes = np.array(ndimage.sum(array, id_regions, range(num_ids + 1)))
    area_mask = (id_sizes == 1)
    filtered_array[area_mask[id_regions]] = 0
    return filtered_array


def rebin(a, bs, func=np.sum):
    """Rebin array

    Parameters:
    -----------
    a: Ndarray (2D or 3D)
        Input array (or stack of arrays: N x Nx x Ny)
    bs: Tuple (nx, ny)
        Size of the bin applied, i.e. (2, 2) means that the output array will
        be reduced by a factor of 2 in both directions.
        If bs is None the original array is returned.
    func: Function
        Function applied to the individual bins [np.sum]. Alternatively,
        np.mean could be used.

    Returns:
    --------
        Rebinned ndarray

    Notes:
    ------
        + The bin size must be compatible with the size of the input array.
        + When used with maps or images, please note that the calibration
          (dc.nm_per_pixels) will be too small by the corresponding rebin
          factor.

    Examples:
    ---------

    >>> import JEOL_eds.utils as JU

    >>> a = np.arange(36).reshape(6, 6)
    >>> rebinned = JU.rebin(a, (2, 3))
    >>> rebinned
    array([[ 24,  42],
           [ 96, 114],
           [168, 186]])

    >>> rebinned.sum() == a.sum()
    np.True_

    >>> a.mean() == (JU.rebin(a, (3, 3), func=np.mean).mean())
    np.True_

    Try with 3D array
    >>> a = np.asarray(range(24)).reshape(3, 4, 2)
    >>> a.shape
    (3, 4, 2)

    >>> rebinned = rebin(a, (2, 1))
    >>> rebinned.shape
    (3, 2, 2)

    >>> a[0, 2:4, 0].sum() == rebinned[0, 1, 0]
    np.True_
    """
    if bs is None:  # No rebinning required
        return a

    assert len(a.shape) in (2, 3)
    assert len(bs) == 2

    assert (a.shape[-1] / bs[-1]).is_integer()
    assert (a.shape[-2] / bs[-2]).is_integer()

    if len(a.shape) == 3:
        N = a.shape[0]  # number of images in 3D array
        a = a.reshape(a.shape[-2] * N, a.shape[-1])
        r = func(a.reshape(a.shape[0] // bs[0], bs[0],
                           a.shape[1] // bs[1], bs[1]),
                 axis=(1, 3))
        return r.reshape(N, r.shape[0] // N, r.shape[1])
    else:
        return func(a.reshape(a.shape[0] // bs[0], bs[0],
                              a.shape[1] // bs[1], bs[1]),
                    axis=(1, 3))


def __plot_line(x, y,
                outfile=None,
                x_label=None, y_label=None,
                M_ticks=None,
                log_y=None,
                **kws):
    """Line plot y=f(x)

    Parameters
    ----------
    x : Ndarray
        x data
    y : Ndarray
        y data
    x_label : Str
        Label for x-axis
    y_label : Str
        Label for y-axis
    M_ticks : Tuple (int, int)
        Number of minor ticks used for x and y axis. If you want to plot minor
        ticks for a single axis, use None for other axis. Parameter for y axis
        is ignored in logarithmic plots.
    log_y : Bool
        Plot linear [Default] or logarithmic y-axis.
    outfile : Str
        Plot is saved as `outfile`. Graphics file type is inferred from
        extension. Available formats might depend on your installation.
    **kws
        Additional keywords passed to ``matplotlib.pyplot.plot()``

   """
    if outfile:
        ext = os.path.splitext(outfile)[1][1:].lower()
        supported = plt.figure().canvas.get_supported_filetypes()
        assert ext in supported

    plt.plot(x, y, **kws)
    ax = plt.gca()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if log_y:
        ax.set_yscale('log')
    # Plot minor ticks on the axis required. Careful: matplotlib specifies the
    # number of intervals which is one more than the number of ticks!
    if M_ticks is not None:
        mx, my = M_ticks
        if mx is not None:
            ax.xaxis.set_minor_locator(AutoMinorLocator(mx + 1))
        if my is not None:
            ax.yaxis.set_minor_locator(AutoMinorLocator(my + 1))

    if outfile:
        plt.savefig(outfile)


def __scalebar_length(label):
    """Returns length [nm] extracted from label string

    Parameters
    ----------
    label : Str
        Label string (for scale bar) as 'NNN U' with or without space between
        number (NNN) and unit (U). Allowed units are 'nm', 'μm' ('um'),
        'Å' ('A') or 'px' for uncalibrated images.

    Returns
    --------
    length : Float
        Length (of scale bar) [nm].
    """
    if label.endswith('nm') or label.endswith('px'):
        length = float(label[:-2])
    elif label.endswith('μm'):
        length = float(label[:-2]) * 1000.0     # convert no nm
    elif label.endswith('Å'):
        length = float(label[:-1]) / 10.0       # convert to nm
    else:
        length = None
    return length


def __add_scalebar(ax, scale_bar, extent):
    """Adds scale bar to plot at axes ``ax``.

    Parameters
    ----------
    ax : Axes instance
    scale_bar :  Dict
    extent : List [0, width, 0, height]

    """
    if (
        isinstance(scale_bar, dict)
        and 'f_calib' in scale_bar
        and 'label' in scale_bar
        and (
            scale_bar['label'].endswith('px')
            or scale_bar['label'].endswith('nm')
            or scale_bar['label'].endswith('μm')
            or scale_bar['label'].endswith('um')
            or scale_bar['label'].endswith('A')
            or scale_bar['label'].endswith('Å')
        )
    ):
        if scale_bar['label'].endswith('A'):    # 'A' can be used as synonym for 'Å'
            scale_bar['label'] = scale_bar['label'][:-1] + 'Å'
        if scale_bar['label'].endswith('um'):    # 'um' can be used as synonym for 'μm'
            scale_bar['label'] = scale_bar['label'][:-2] + 'μm'
        if scale_bar['label'].endswith('px'):   # 'px' means uncalibrated i.e. set 'f_calib'
            scale_bar['f_calib'] = 1.0

        pos = scale_bar['position'] if 'position' in scale_bar else 'lower right'
        color = scale_bar['color'] if 'color' in scale_bar else 'black'
        fontprops = fm.FontProperties(size=16)
        length = __scalebar_length(scale_bar['label'])

        if scale_bar['f_calib'] is None:    # Uncalibrated, use 'px'.
            if scale_bar['label'].endswith('Å'):
                label = scale_bar['label'][:-1] + 'px'
            else:
                label = scale_bar['label'][:-2] + 'px'
        else:
            label = scale_bar['label']
        scalebar = AnchoredSizeBar(ax.transData,
                                   length,
                                   label,
                                   pos,
                                   pad=0.5,
                                   color=color,
                                   frameon=False,
                                   size_vertical=extent[3] * 0.01,
                                   fontproperties=fontprops)
        ax.add_artist(scalebar)


def __get_extent(m, scale_bar):
    """Returns extent in data coordinates or image pixels (scale bar not defined)

    Parameters
    ----------
    m : Ndarray
        Image or map.
    scale_bar : Dict
        Scale bar information (calibration factor [nm/pixel]).

    Returns
    -------
        extent : List [0, width, 0, height]
    """
    if (
        isinstance(scale_bar, dict)
        and 'f_calib' in scale_bar
        and 'label' in scale_bar
    ):
        width = scale_bar['f_calib'] * m.shape[0]
        height = scale_bar['f_calib'] * m.shape[1]
    else:
        width = m.shape[0]
        height = m.shape[1]
    return [0, width, 0, height]


def create_overlay(images, colors,
                   legends=None, BG_image=None, outfile=None, scale_bar=None):
    """Plots overlay of `images` with `colors`.

    Parameters
    ----------
    images :  List or tuple
        Images to be overlaid (must be of identical shape).
    colors : List or tuple.
        List of colors used to overlay the images provided in the same order as
        `images`. Surplus colors provided are ignored.
    legends : List or tuple.
        List of legends used to annotate individual maps in overlay.
    BG_image : Ndarray.
        Background image (gray scale). Must be of same shape as `images`.
    outfile : Str
        Plot is saved as `outfile`. Graphics file type is inferred from
        extension. Available formats might depend on your installation.
    scale_bar : Dict
        Optional dict that defines scale bar to be inserted into map. Allows
        for the following keys
        label : Str
            Label (NNN U) for scale bar (required) with NNN number and U units
            ['nm'|'μm' or 'um'|'Å' or 'A'].
        f_calib : Float
            Calibration (pixel size in nm) (required).
        position : Str
            Position of scale bar ['lower left']. Valid positions are
            ('upper left', 'upper center', 'upper right', 'center left',
                 'center', 'center right', 'lower left','lower center,
                 'lower right').
        color: Str
            Color of scale bar and label ['black'].

    Notes
    -----
        `images` are overlaid in the sequence given. Thus fully saturated
        pixels of only the last image given are visible. Therefore the final
        figure depends on the order of the images given with the most important
        image last.

    Examples
    --------
    >>> from JEOL_eds import JEOL_pts
    >>> import JEOL_eds.utils as JU

    Load data.
    >>> dc = JEOL_pts('data/64.pts', read_drift="yes")

    Extract some elemental maps. Where possible, add contribution of several
    lines.
    >>> Al = dc.map(interval=(1.4, 1.6), energy=True)

    >>> Si = dc.map(interval=(1.7, 1.8), energy=True)

    >>> Cu = dc.map(interval=(7.9, 8.2), energy=True)

    >>> Cu += dc.map(interval=(8.8, 9.1), energy=True)

    Create overlays. No legends plotted and
    file is saved.
    >>> JU.create_overlay((Si, Al), ('Red', 'Green'),
    ...                   outfile='test.pdf')

    Visualize the copper background.

    >>> JU.create_overlay([Cu],
    ...                   ['Maroon'],
    ...                   legends=['Cu'],
    ...                   BG_image=dc.drift_images[0]) # doctest: +IGNORE_EXCEPTION_DETAIL

    Add scale bar at default position (lower right) and default color (black).
    >>> scale_bar = {'label': '100 px',
    ...             'f_calib': dc.nm_per_pixel}

    >>> JU.create_overlay([Cu],
    ...                   ['Maroon'],
    ...                   legends=['Cu'],
    ...                   BG_image=dc.drift_images[0],
    ...                   scale_bar=scale_bar)
    """
    assert isinstance(images, (list, tuple))
    assert isinstance(colors, (list, tuple))
    assert len(colors) >= len(images)   # Sufficient colors provided
    assert all(image.shape == images[0].shape for image in images)    # all same shape
    if legends:
        assert isinstance(legends, (list, tuple))
        assert len(legends) == len(images)
    if BG_image is not None:
        assert images[0].shape == BG_image.shape
    if outfile:
        ext = os.path.splitext(outfile)[1][1:].lower()
        supported = plt.figure().canvas.get_supported_filetypes()
        assert ext in supported

    # Obtain size (w x h) of image
    extent = __get_extent(images[0], scale_bar)

    # Show background image
    if BG_image is not None:
        plt.imshow(BG_image, cmap='gist_gray', extent=extent)

    # Create overlays. Use fake image `base` with fully saturated color and
    # use real image as alpha channel (transparency)
    base = np.ones_like(images[0])
    for image, color in zip(images, colors):
        # Custom color map that contains only `color` at full saturation
        cmap = LinearSegmentedColormap.from_list("cmap", (color, color))
        alpha = image / image.max()
        plt.imshow(base, cmap=cmap, alpha=alpha,
                   vmin=0, vmax=1, extent=extent)

    # Fine tune plot
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])

    # Add legends. Position and font size depends on image size
    if legends:
        fontsize = 12
        delta = extent[3] // fontsize  # Found by trial-and-error
        for i in range(len(images)):
            ax.text(extent[1], extent[3] - i * delta, legends[i],
                    size=fontsize,
                    color=colors[i], backgroundcolor='white')

    __add_scalebar(ax, scale_bar, extent)

    if outfile:
        plt.savefig(outfile)


def __make_cmap(color, gamma=1.0, background="white"):
    """Return color map with uniform gradient from white to `color`.

    Parameters
    ----------
    color: Str
        Named color (or matplotlib.colormap). If a color map is supplied it is
        returned unmodified.
    gamma : Float
        Default [1.0]: No transformation applied ``gamma`` > 1.0: Reaches high
        saturation already at low input values while ``gamma`` < 1.0 only
        reaches high saturation at high input values.
    background : Str ["white" | "black"].
        Custom color map is made spanning ``background`` to ``color``.

    Returns
    -------
        color map
    """
    if background.lower() not in ["white", "black"]:
        raise ValueError(f"Only 'white' and 'black' allowed as background, not '{background}'.")
    start = 1 if background.lower() == "white" else 0

    try:    # Is ``color`` a named color?
        rgba = to_rgba(color)
    except ValueError:
        return color    # ``color`` is already a color map
    N = 256
    t_color = np.ones((N, 4))   # alpha channel [:,3] is set to 1
    for i in range(3):          # set rgb channels
        t_color[:, i] = np.linspace(start, rgba[i], N) ** gamma
    return ListedColormap(t_color)


def plot_map(m, color,
             label=None,
             gamma=1.0,
             outfile=None,
             background="white",
             remove_outliers=False,
             smooth=None,
             scale_bar=None):
    """Nicely plots map.

    Parameters
    ----------
    m : Ndarray
        Map to be plotted.
    color : Str or tuple
        Color (name or rgb-tuple) used to generate color map. Color map will
        span between ``background`` and ``color``. If the name of a matplotlib
        color map is supplied it will be used and ``background`` is ignored.
    label : Str
        Optional label to be plotted on map.
    gamma : Float
        Optional value to make a non-linear color map. Only used for custom
        color maps i.e. if ``color`` does not represent a matplotlib color map.
        ``gamma`` > 1.0: Reaches high saturation already at low values of ``m``
        while ``gamma`` < 1.0 only reaches high saturation at high values.
    outfile : Str
        Optional name of an output file. Any format supported by matplotlib can
        be used. If none is supplied, output goes to screen.
    background : Str
        Optional background color ("white" or "black") only used for custom
        color maps i.e. if ``color`` does not represent a matplotlib color map.
    remove_outliers : Bool
        Isolated (no neighbors) pixels in map will be removed if set to True.
    smooth : Float
        If supplied, map will be smoothed by 2D Gaussian with sigma=``smooth``.
        FWHH of Gaussian corresponds to 2.355*``sigma``.
    scale_bar : Dict
        Optional dict that defines scale bar to be inserted into map. Allows
        for the following keys
        label : Str
            Label (NNN U) for scale bar (required) with NNN number and U units
            ['nm'|'μm' or 'um'|'Å' or 'A'].
        f_calib : Float
            Calibration (pixel size in nm) (required).
        position : Str
            Position of scale bar ['lower left']. Valid positions are
            ('upper left', 'upper center', 'upper right', 'center left',
                 'center', 'center right', 'lower left','lower center,
                 'lower right').
        color: Str
            Color of scale bar and label ['black'].

    Examples
    --------
    >>> from JEOL_eds import JEOL_pts
    >>> import JEOL_eds.utils as JU

    Load data:
    >>> dc = JEOL_pts('data/64.pts')

    Map of total X-ray intensity:
    >>> m = dc.map()

    Plot nice map using custom color map black to purple. ``gamma=0.9``
    enhances details. Little smoothing (FWHH=1.75 pixels) is applied:
    >>> JU.plot_map(m,
    ...             "purple",
    ...             label="Itot",
    ...             background="black",
    ...             gamma=0.9,
    ...             smooth=0.75)

    Specify ``color`` as rgb tuple (range 0 <= value <=255):
    >>> JU.plot_map(m,
    ...             (250, 136, 37),
    ...             label="Itot",
    ...             background="black",
    ...             gamma=0.9,
    ...             smooth=0.75)

    Same as before but rgb tuple (range 0 <= value <=1):
    >>> JU.plot_map(m,
    ...             (0.98, 0.53, 0.15),
    ...             label="Itot",
    ...             background="black",
    ...             gamma=0.9,
    ...             smooth=0.75)

    Use predefined color map "Blues". ``background`` and ``gamma`` are ignored:
    >>> JU.plot_map(m,
    ...             "Blues",
    ...             label="Itot",
    ...             background="black",
    ...             gamma=0.9,
    ...             smooth=0.75)

    Insert scale bar. Use 'A' as synonym for 'Å':
    >>> scale_bar={'label': '500 A',
    ...            'f_calib': dc.nm_per_pixel,
    ...            'position': 'lower left',
    ...            'color': 'white'}

    >>> JU.plot_map(m,
    ...             'inferno',
    ...             scale_bar=scale_bar,
    ...             label="a)")
    """
    if outfile:
        ext = os.path.splitext(outfile)[1][1:].lower()
        supported = plt.figure().canvas.get_supported_filetypes()
        assert ext in supported

    if remove_outliers:     # remove stray (single isolated) pixels
        m = filter_isolated_pixels(m)

    if smooth:
        m = ndimage.gaussian_filter(m, smooth)

    # If a rgb-tuple is specified as e.g. (250, 176, 37)
    if (
        isinstance(color, tuple)
        and max(color) > 1.0
        and all(0 <= val < 256 for val in color)
    ):
        color = tuple(val / 256 for val in color)

    # Obtain size (w x h) of image
    extent = __get_extent(m, scale_bar)

    cmap = __make_cmap(color, gamma=gamma, background=background)
    plt.imshow(m, cmap=cmap, extent=extent)
    plt.colorbar(label="counts  [-]")
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])

    if label:
        # Ensure that legend is visible black in white box on black BG or vice versa
        label_BGcolor = "black" if background.lower() == "white" else "white"
        label_color = "black" if label_BGcolor == "white" else "white"
        # Position to print label in data coordinates found by trial-and-error.
        ax.text(extent[0] * 0.05, extent[3] * 0.85,
                label,
                size=24,
                color=label_color,
                backgroundcolor=label_BGcolor)

    __add_scalebar(ax, scale_bar, extent)

    if outfile:
        plt.savefig(outfile)


def plot_spectrum(s, E_range=None, M_ticks=None,
                  log_y=False, outfile=None, **kws):
    """Plots a nice spectrum.

    Parameters
    ----------
    s : Ndarray
        Spectral data which is expected to cover the energy range
        0.0 < E <= E_max at an resolution of 0.01 keV per data point.
    E_range : Tuple (E_low, E_high).
        Energy range to be plotted.
    M_ticks : Tuple (mx, my)
        Number of minor ticks used for x and y axis. If you want to plot minor
        ticks for a single axis, use None for other axis. Parameter for y axis
        is ignored in logarithmic plots.
    log_y : Bool
        Plot linear [Default] or logarithmic y-axis.
    outfile : Str
        Plot is saved as `outfile`. Graphics file type is inferred from
        extension. Available formats might depend on your installation.
    **kws
        Additional keywords passed to ``matplotlib.pyplot.plot()``

    Examples
    --------
    >>> from JEOL_eds import JEOL_pts
    >>> import JEOL_eds.utils as JU

    Load data.
    >>> dc = JEOL_pts('data/128.pts', only_metadata=True)

    Plot full reference spectrum with logarithmic y-axis:
    >>> JU.plot_spectrum(dc.ref_spectrum, log_y=True)

    Plot and save reference spectrum between 1.0 and 2.5 keV. Plot one minor
    tick on x-axis and four on y-axis. Pass some keywords to
    `matplotlib.pyplot.plot()`:
    >>> JU.plot_spectrum(dc.ref_spectrum,
    ...                  E_range=(1, 2.5),
    ...                  M_ticks=(1, 4),
    ...                  outfile='ref_spectrum.pdf',
    ...                  color='Red', linestyle='-.', linewidth=1.0)
    """
    F = 1 / 100     # Calibration factor (Energy per channel)

    if E_range is not None:
        E_low, E_high = E_range
        if E_high > s.shape[0] * F:  # E_high is out of range
            E_high = s.shape[0] * F
    else:
        E_low, E_high = 0, s.shape[0] * F

    N = int(np.round((E_high - E_low) / F))    # Number of data points
    x = np.linspace(E_low, E_high, N)   # Energy axis
    # Indices corresponding to spectral interval
    i_low = int(np.round(E_low / F))
    i_high = int(np.round(E_high / F))

    __plot_line(x, s[i_low:i_high],
                x_label='E  [keV]', y_label='counts  [-]',
                M_ticks=M_ticks, log_y=log_y, outfile=outfile,
                **kws)


def export_spectrum(s, outfile, E_range=None):
    """Exports spectrum as tab delimited ASCII.

    Parameters
    ----------
    s : Ndarray.
        Spectral data which is expected to cover the energy range
        0.0 < E <= E_max at an resolution of 0.01 keV per data point.
    outfile : Str.
        Filename where data is saved.
    E_range : Tuple (E_low, E_high).
        Limit data to optional energy range.

    Examples
    --------
    >>> from JEOL_eds import JEOL_pts
    >>> import JEOL_eds.utils as JU

    Load data.
    >>> dc = JEOL_pts('data/64.pts', only_metadata=True)

    Export full reference spectrum as 'test_spectrum.dat':
    >>> JU.export_spectrum(dc.ref_spectrum, 'test_spectrum.dat')

    Only export data between 1.0 and 2.5 keV:
    >>> JU.export_spectrum(dc.ref_spectrum, 'test_spectrum.dat',
    ...                    E_range=(1, 2.5))
    """
    F = 1 / 100     # Calibration factor (Energy per channel)

    if E_range is not None:
        E_low, E_high = E_range
        if E_high > s.shape[0] * F:  # E_high is out of range
            E_high = s.shape[0] * F
    else:
        E_low, E_high = 0, s.shape[0] * F

    N = int(np.round((E_high - E_low) / F))    # Number of data points
    data = np.zeros((N, 2))
    data[:, 0] = np.linspace(E_low, E_high, N)   # Energy axis
    # Indices corresponding to spectral interval
    i_low = int(np.round(E_low / F))
    i_high = int(np.round(E_high / F))
    data[:, 1] = s[i_low:i_high]    # copy desired range

    header = '# E [keV]        counts [-]'
    fmt = '%.2f\t%d'
    np.savetxt(outfile, data, header=header, fmt=fmt)


def plot_tseries(ts, M_ticks=None, outfile=None, **kws):
    """Plots a nice time series.

    Parameters
    ----------
    ts : Ndarray.
        Time series (integrated x-ray intensities.
    M_ticks : Tuple (mx, my).
        Number of minor ticks used for x and y axis. If you want to plot minor
        ticks for a single axis, use None for other axis.
    outfile : Str.
        Plot is saved as `outfile`. Graphics file type is inferred from
        extension. Available formats might depend on your installation.
    **kws
        Additional keywords passed to ``matplotlib.pyplot.plot()``

    Examples
    --------
    >>> from JEOL_eds import JEOL_pts
    >>> import JEOL_eds.utils as JU

    Load data:
    >>> dc = JEOL_pts('data/128.pts', split_frames=True)

    Get integrated x-ray intensity for carbon Ka peak but exclude frames 11
    and 12):
    >>> frames = list(range(dc.dcube.shape[0]))

    >>> frames.remove(11)

    >>> frames.remove(12)

    >>> ts = dc.time_series(interval=(0.22, 0.34), energy=True, frames=frames)

    Plot and save time series with some keywords passed to
    `matplotlib.pyplot.plot()`.
    >>> JU.plot_tseries(ts,
    ...                 M_ticks=(9, 4),
    ...                 outfile='carbon_Ka.pdf',
    ...                 color='Red', linestyle='-.', linewidth=1.0)
    """
    __plot_line(range(ts.shape[0]), ts,
                x_label='Frame Index  [-]', y_label='counts  [-]',
                M_ticks=M_ticks, outfile=outfile,
                **kws)


def export_tseries(ts, outfile):
    """Export time series as tab delimited ASCII.

    Parameters
    ----------
    ts : Ndarray.
        Time series (integrated x-ray intensities.
    outfile : Str.
        Data is saved as `outfile`.

    Examples
    --------
    >>> from JEOL_eds import JEOL_pts
    >>> import JEOL_eds.utils as JU

    Load data:
    >>> dc = JEOL_pts('data/128.pts', split_frames=True)

    Get integrated x-ray intensity for carbon Ka peak but exclude frames 11
    and 12):
    >>> frames = list(range(dc.dcube.shape[0]))

    >>> frames.remove(11)

    >>> frames.remove(12)

    >>> ts = dc.time_series(interval=(0.22, 0.34), energy=True, frames=frames)

    Export time series:
    >>> JU.export_tseries(ts, 'test_tseries.dat')
    """
    N = ts.shape[0]
    data = np.zeros((N, 2))
    data[:, 0] = range(N)
    data[:, 1] = ts

    header = '# Frame idx [-]        counts [-]'
    fmt = '%d\t%f'
    np.savetxt(outfile, data, header=header, fmt=fmt)


def plot_profile(x, y, units='px', M_ticks=None, outfile=None, **kws):
    """Plots a nice profile.

    Parameters
    ----------
    x : Ndarray.
        x values.
    y : Ndarray
        y values.
    units : Str
        Units (length) of x data. The string supplied is only used to generate
        the axis label.
    M_ticks : Tuple (mx, my).
        Number of minor ticks used for x and y axis. If you want to plot minor
        ticks for a single axis, use None for other axis.
    outfile : Str.
        Plot is saved as `outfile`. Graphics file type is inferred from
        extension. Available formats might depend on your installation.
    **kws
        Additional keywords passed to ``matplotlib.pyplot.plot()``

    Examples
    --------
    >>> from JEOL_eds import JEOL_DigiLine
    >>> import JEOL_eds.utils as JU

    Read data:
    >>> dl = JEOL_DigiLine('data/DigiLine/View000_0000003.pts')

    Extract Oxygen profile with x axis in [nm]:
    >>> x, p_O = dl.profile(interval=(0.45, 0.6),
    ...                     energy=True, xCalib=True)

    >>> JU.plot_profile(x, p_O,
    ...                 units='nm',
    ...                 M_ticks=(9, 4),
    ...                 color='Red', linestyle='-.', linewidth=1.0)
    """
    x_label = f'Length  [{units}]'
    __plot_line(x, y,
                x_label=x_label, y_label='counts  [-]',
                M_ticks=M_ticks, outfile=outfile,
                **kws)


def export_profile(x, y, outfile, units='px'):
    """Export profile as tab delimited ASCII.

    Parameters
    ----------
    x : Ndarray.
        x values.
    y : Ndarray
        y values.
    outfile : Str.
        Plot is saved as `outfile`. Graphics file type is inferred from
        extension. Available formats might depend on your installation.
    units : str
        Units (length) of x data. The string supplied is only used to
        generate the header comment.

    Examples
    --------
    >>> from JEOL_eds import JEOL_DigiLine
    >>> import JEOL_eds.utils as JU

    Read data:
    >>> dl = JEOL_DigiLine('data/DigiLine/View000_0000003.pts')

    Extract Oxygen profile with x axis in [nm]:
    >>> x, p_O = dl.profile(interval=(0.45, 0.6),
    ...                     energy=True, xCalib=True)

    Export data:
    >>> JU.export_profile(x, p_O, 'test_tseries.dat', units='nm')
    """
    assert x.shape[0] == y.shape[0]

    header = f'# Position [{units}]        counts [-]'
    fmt = '%d\t%f'
    np.savetxt(outfile, np.vstack((x, y)).T, header=header, fmt=fmt)


def __linewidth_from_data_units(linewidth, axis):
    """Convert a linewidth in pixels to points.

    Parameters
    ----------
    linewidth : float
        Line width in pixels.
    axis :  matplotlib axis
        The axis which is used to extract the relevant transformation data
        (data limits and size must not change afterwards).

    Returns
    -------
    linewidth : float
        Line width in points.

    Notes
    -----
        Adapted from https://stackoverflow.com/questions/19394505
    """
    fig = axis.get_figure()
    length = fig.bbox_inches.width * axis.get_position().width
    value_range = np.diff(axis.get_xlim())[0]
    # Convert length to points
    length *= 72    # 72 points per inch
    # Scale linewidth to value range
    return linewidth * (length / value_range)


def show_line(image, line, linewidth=1, outfile=None, **kws):
    """Plots a white (profile) line on image.

    Parameters
    ----------
    image : Ndarray
        Image onto which the line will be plotted.
    line : Tuple (int, int, int, int).
        Defines line (start_v, start_h, stop_v, stop_h).
    linewidth : Int
        Width (pixels) of line to be drawn.
    outfile:  Str
        Optional filename where plot is saved.

    Notes
    -----
    **kws are only applied to the image plot and not to the line.

    Examples
    --------
    >>> from JEOL_eds import JEOL_pts
    >>> import JEOL_eds.utils as JU

    Load data:
    >>> dc = JEOL_pts('data/128.pts')

    Carbon map:
    >>> C_map = dc.map(interval=(0.22, 0.34), energy=True)

    Define line (10 pixels wide) and verify definition:
    >>> line = (80, 5, 110, 100)

    >>> width = 10

    >>> JU.show_line(C_map, line, linewidth=width, cmap='inferno')
    """
    if outfile:
        ext = os.path.splitext(outfile)[1][1:].lower()
        supported = plt.figure().canvas.get_supported_filetypes()
        assert ext in supported

    ax = plt.imshow(image, **kws)
    x = (line[1], line[3])
    y = (line[0], line[2])
    width = __linewidth_from_data_units(linewidth, ax.axes)
    plt.plot(x, y, color='white', linewidth=width)

    if outfile:
        plt.savefig(outfile)


def get_profile(image, line, linewidth=1):
    """Returns a profile along line on image.

    Parameters
    ----------
    image : Ndarray.
        Image onto which the line will be plotted.
    line : Tuple (int, int, int, int).
        Defines line (start_v, start_h, stop_v, stop_h).
    linewidth : Int.
        Width of profile line (to be integrated).

    Returns
    -------
    profile : Ndarray
        Profile, length unit is pixels (as in image).

    Examples
    --------
    >>> from JEOL_eds import JEOL_pts
    >>> import JEOL_eds.utils as JU

    Load data:
    >>> dc = JEOL_pts('data/128.pts')

    Carbon map:
    >>> C_map = dc.map(interval=(0.22, 0.34), energy=True)

    Define line and verify definition:
    >>> line = (80, 5, 110, 100)

    >>> width = 10

    >>> JU.show_line(C_map, line, linewidth=width, cmap='inferno')

    Calculate profile along the line (width equals 10 pixels) and plot it:
    >>> profile = JU.get_profile(C_map, line, linewidth=width)

    >>> import matplotlib.pyplot as plt

    >>> ax = plt.plot(profile)
    """
    profile = profile_line(image,
                           line[0:2], line[2:],
                           linewidth=linewidth,
                           mode='nearest')
    return profile


def show_ROI(image, ROI, outfile=None, alpha=0.4, **kws):
    """Plots ROI on image.

    Parameters
    ----------
    image : Ndarray
        Image where ROI will be applied.
    ROI : Tuple
        If tuple is (int, int) ROI defines a point (h, v).
        If tuple is (int, int, int) ROI defines a circle (h, v, r) .
        If tuple is (int, int, int, int) ROI defines rectangle (t, b, l, r).
    outfile : Str
        Optional filename where plot is saved.
    alpha : Float
        Transparency used to draw background image for ROI [0.4].

    Examples
    --------
    >>> from JEOL_eds import JEOL_pts
    >>> import JEOL_eds.utils as JU

    Load data and create map of total x-ray intensity.
    >>> dc = JEOL_pts('data/complex_oxide.h5')
    >>> my_map = dc.map()

    Verify definition of rectangular ROI. Make image more visible (less
    transparent).
    >>> JU.show_ROI(my_map, (50, 250, 10, 75), alpha=0.6)

    Check the definition of a circular ROI first.
    Override the default color map.
    >>> JU.show_ROI(my_map, (270, 122, 10), cmap='inferno')
    """
    if len(ROI) == 2:
        im = image.copy().astype('float')
        im[ROI[0], :] = np.nan
        im[:, ROI[1]] = np.nan
        plt.imshow(im, **kws)
    elif len(ROI) == 3:
        x, y = np.ogrid[:image.shape[0], :image.shape[1]]
        r = np.sqrt((x - ROI[0])**2 + (y - ROI[1])**2)
        mask = r <= ROI[2]
        plt.imshow(image * mask, **kws)         # ROI
        plt.imshow(image, alpha=alpha, **kws)     # transparent image
    elif len(ROI) == 4:
        mask = np.full_like(image, False, dtype='bool')
        mask[ROI[0]:ROI[1], ROI[2]:ROI[3]] = True
        plt.imshow(image * mask, **kws)         # ROI
        plt.imshow(image, alpha=alpha, **kws)     # transparent image
    else:
        raise ValueError(f'Invalid ROI {ROI}.')

    if outfile:
        plt.savefig(outfile)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
