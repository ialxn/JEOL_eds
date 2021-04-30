#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 15:11:53 2021

@author: alxneit
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator

def create_overlay(images, colors, legends=None, BG_image=None, outfile=None):
    """Plots overlay of `images` with `colors`.

        Parameters
        ----------
           images:  List or tuple.
                    Images to be overlaid (must be of identical shape).
           colors:  List or tuple.
                    List of colors used to overlay the images provided
                    in the same order as `images`. Surplus colors provided
                    are ignored.
          legends:  List or tuple.
                    List of legends used to annotate individual maps in
                    overlay.
         BG_image:  Ndarray.
                    Background image (grey scale). Must be of same shape as
                    `images`.
          outfile:  Str.
                    Plot is saved as `outfile`. Graphics file type is inferred
                    from extension. Available formats might depend on your
                    installation.

        Notes
        -----
                `images` are overlaid in the sequence given. Thus fully saturated
                pixels of only the last image given are visible. Therefore the
                final figure depends on the order of the images given with the
                most important image last.

        Examples
        --------
        >>>> from JEOL_eds import JEOL_pts
        >>>> from JEOL_eds.utils import create_overlay

        # Load data.
        >>>> dc = JEOL_pts('test/SiFeO.pts', E_cutoff=8.5, read_drift=True)

        # Extract elemental maps. Add contribution of all available lines.
        >>>> Fe = dc.map(interval=(6.2, 7.25), energy=True)  # Ka,b
        >>>> Fe += dc.map(interval=(0.65, 0.8), energy=True)     # La,b
        >>>> Si = dc.map(interval=(1.65, 1.825), energy=True)   # Ka,b
        >>>> O = dc.map(interval=(0.45, 0.6), energy=True)  # Ka,b

        # Create overlay. Oxygen is hardly visible as it covered by silicon and
        # iron. Focus is on iron distribution. No legends plotted.
        # File is saved
        >>>> create_overlay((O, Si, Fe), ('Red', 'Green', 'Blue'),
                            outfile='test.pdf')

        # Focus on oxygen. Follows both, iron and silicon distributions.
        >>>> create_overlay([Fe, Si, O],
                            ['Blue', 'Red', 'Green'],
                            legends=['Fe', 'Si', 'O'])

        # FeOx distribution using first of the `drift_images` as background
        >>>> create_overlay([Fe, O],
                            ['Red', 'Blue'],
                            legends=['Fe', 'O'],
                            BG_image=dc.drift_images[0])
    """
    assert isinstance(images, (list, tuple))
    assert isinstance(colors, (list, tuple))
    assert len(colors) >= len(images)   # Sufficient colors provided
    assert all(image.shape==images[0].shape for image in images)    # all same shape
    if legends:
        assert isinstance(legends,(list, tuple))
        assert len(legends) == len(images)
    if BG_image is not None:
        assert images[0].shape == BG_image.shape
    if outfile:
        ext = os.path.splitext(outfile)[1][1:].lower()
        supported = plt.figure().canvas.get_supported_filetypes()
        assert ext in supported

    # Show background image
    if BG_image is not None:
        plt.imshow(BG_image, cmap='gist_gray')
    # Create overlays. Use fake image `base` with fully saturated color and
    # use real image as alpha channel (transparency)
    base = np.ones_like(images[0])
    for image, color in zip(images, colors):
        # Custom color map that contains only `color` at full saturation
        cmap = LinearSegmentedColormap.from_list("cmap", (color, color))
        alpha = image / image.max()
        plt.imshow(base, cmap=cmap, alpha=alpha,
                   vmin=0, vmax=1)

    # Fine tune plot
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])

    # Add legends. Position and font size depends on image size
    if legends:
        isize = images[0].shape[0]
        fontsize = isize // 12
        x = isize + fontsize
        for i in range(len(images)):
            y = i * fontsize
            ax.text(x, y, legends[i],
                    size=fontsize,
                    color=colors[i], backgroundcolor='white')

    if outfile:
        plt.savefig(outfile)


def plot_spectrum(s, E_range=None, M_ticks=None,
                  log_y=False, outfile=None, **kws):
    """Plots a nice spectrum

        Parameters
        ----------
                s:  Ndarray.
                    Spectral data which is expected to cover the energy range
                    0.0 < E <= E_max at an resolution of 0.01 keV per data point.
          E_range:  Tuple (E_low, E_high).
                    Energy range to be plotted.
          M_ticks:  Tuple (mx, my).
                    Number of minor ticks used for x and y axis. If you want to
                    plot minor ticks for a single axis, use None for other axis.
                    Parameter for y axis is ignored in logarithmic plots.
            log_y:  Bool.
                    Plot linear or logarithmic y-axis.
          outfile:  Str.
                    Plot is saved as `outfile`. Graphics file type is inferred
                    from extension. Available formats might depend on your
                    installation.

        Examples
        --------
        >>>> from JEOL_eds import JEOL_pts
        >>>> from JEOL_eds.utils import plot_spectrum

        # Load data.
        >>>> dc = JEOL_pts('test/SiFeO.pts', E_cutoff=8.5)

        # Plot full reference spectrum with logaritmic y-axis.
        >>>> plot_spectrum(dc.ref_spectrum, log_y=True)

        # Plot and save reference spectrum between 1.0 and 2.5 keV.
        # Plot one minor tick on x-axis and four on y-axis. Pass
        # some keywords to `matplotlib.pyplot.plot()`.
        >>>> plot_spectrum(dc.ref_spectrum,
                           E_range=(1, 2.5),
                           M_ticks=(1,4),
                           outfile='ref_spectrum.pdf',
                           color='Red', linestyle='-.', linewidth=1.0)
    """
    F = 1/100     # Calibration factor (Energy per channel)
    if outfile:
        ext = os.path.splitext(outfile)[1][1:].lower()
        supported = plt.figure().canvas.get_supported_filetypes()
        assert ext in supported

    if E_range is not None:
        E_low, E_high = E_range
        if E_high > s.shape[0] * F: # E_high is out of range
            E_high = s.shape[0] * F
    else:
        E_low, E_high = 0, s.shape[0] * F

    N = int(np.round((E_high - E_low) / F))    # Number of data points
    x = np.linspace(E_low, E_high, N)   # Energy axis
    # Indices corresponding to spectral interval
    i_low = int(np.round(E_low / F))
    i_high = int(np.round(E_high / F))

    plt.plot(x, s[i_low:i_high], **kws)
    ax = plt.gca()
    ax.set_xlabel('E  [keV]')
    ax.set_ylabel('counts  [-]')

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

def export_spectrum(s, outfile, E_range=None):
    """Exports spectrum as tab delimited ASCII.

        Parameters
        ----------
                s:  Ndarray.
                    Spectral data which is expected to cover the energy range
                    0.0 < E <= E_max at an resolution of 0.01 keV per data point.
          outfile:  Str.
                    Data is saved in `outfile`.
          E_range:  Tuple (E_low, E_high).
                    Energy range to be plotted.

        Examples
        --------
        >>>> from JEOL_eds import JEOL_pts
        >>>> from JEOL_eds.utils import export_spectrum

        # Load data.
        >>>> dc = JEOL_pts('test/SiFeO.pts', E_cutoff=8.5)

        # Export full reference spectrum as 'test_spectrum.dat'.
        >>>> export_spectrum(dc.ref_spectrum, 'test_spectrum.dat')

        # Only export data between 1.0 and 2.5 keV.
        >>>> export_spectrum(dc.ref_spectrum, 'test_spectrum.dat',
                             E_range=(1, 2.5))
    """
    F = 1/100     # Calibration factor (Energy per channel)

    if E_range is not None:
        E_low, E_high = E_range
        if E_high > s.shape[0] * F: # E_high is out of range
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
               ts:  Ndarray.
                    Time series (integrated x-ray intensities.
          M_ticks:  Tuple (mx, my).
                    Number of minor ticks used for x and y axis. If you want to
                    plot minor ticks for a single axis, use None for other axis.
          outfile:  Str.
                    Plot is saved as `outfile`. Graphics file type is inferred
                    from extension. Available formats might depend on your
                    installation.

        Examples
        --------
        >>>> from JEOL_eds import JEOL_pts
        >>>> from JEOL_eds.utils import plot_tseries

        # Load data.
        >>>> dc = JEOL_pts('test/128.pts', split_frames=True)

        # Get integrated x-ray intensity for carbon Ka peak but exclude
        # frames 11 and 12)
        >>>> frames = list(range(dc.dcube.shape[0]))
        >>>> frames.remove(11)
        >>>> frames.remove(12)
        >>>> ts = dc.time_series(interval=(0.22, 0.34), energy=True, frames=frames)

        # Plot and save time series
        # some keywords to `matplotlib.pyplot.plot()`.
        >>>> plot_tseries(ts,
                          M_ticks=(9,4),
                          outfile='carbon_Ka.pdf',
                          color='Red', linestyle='-.', linewidth=1.0)
    """
    if outfile:
        ext = os.path.splitext(outfile)[1][1:].lower()
        supported = plt.figure().canvas.get_supported_filetypes()
        assert ext in supported

    plt.plot(ts, **kws)
    ax = plt.gca()
    ax.set_xlabel('frame index  [-]')
    ax.set_ylabel('counts  [-]')
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

def export_tseries(ts, outfile):
    """Export time series as tab delimited ASCII.
        Parameters
        ----------
               ts:  Ndarray.
                    Time series (integrated x-ray intensities.
          outfile:  Str.
                    Data is saved as `outfile`.

        Examples
        --------
        >>>> from JEOL_eds import JEOL_pts
        >>>> from JEOL_eds.utils import export_tseries

        # Load data.
        >>>> dc = JEOL_pts('test/128.pts', split_frames=True)

        # Get integrated x-ray intensity for carbon Ka peak but exclude
        # frames 11 and 12)
        >>>> frames = list(range(dc.dcube.shape[0]))
        >>>> frames.remove(11)
        >>>> frames.remove(12)
        >>>> ts = dc.time_series(interval=(0.22, 0.34), energy=True, frames=frames)

        # Export time series
        >>>> export_tseries(ts, 'test_tseries.dat')
    """
    N = ts.shape[0]
    data = np.zeros((N, 2))
    data[:, 0] = range(N)
    data[:, 1] = ts

    header = '# Frame idx [-]        counts [-]'
    fmt = '%d\t%d'
    np.savetxt(outfile, data, header=header, fmt=fmt)
