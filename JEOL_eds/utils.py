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

def create_overlay(images, colors, legends=None, outfile=None):
    """Plots overlay of `images` with `colors`.

        Parameters
        ----------
           images:  List or tuple.
                    Images to be overlayed (must be of identical shape).
           colors:  List or tuple.
                    List of colors used to overlay the images provided
                    in the same order as `images`. Surplus colors provided
                    are ignored.
          legends:  List or tuple.
                    List of legends used to annotate individual maps in
                    overlay.
          outfile:  Str.
                    Plot is saved as `outfile`. Graphics file type is inferred
                    from extension. Available formats might depend on your
                    installation.

        Notes
        -----
                `images` are ovelayed in the sequence given. Thus fully saturated
                pixels of only the last image given are visible. Therefore the
                final figure depends on the order of the images given with the
                most important image last.

        Examples
        --------
        >>>> from JEOL_eds import JEOL_pts
        >>>> from JEOL_eds.utils import create_overlay

        # Load data.
        >>>> dc = JEOL_pts('test/SiFeO.pts', E_cutoff=8.5)

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
    """
    assert isinstance(images, (list, tuple))
    assert isinstance(colors, (list, tuple))
    assert len(colors) >= len(images)   # Sufficient colors provided
    assert all(image.shape==images[0].shape for image in images)    # all same shape
    if legends:
        assert isinstance(legends,(list, tuple))
        assert len(legends) == len(images)
    if outfile:
        ext = os.path.splitext(outfile)[1][1:].lower()
        supported = plt.figure().canvas.get_supported_filetypes()
        assert ext in supported

    # Create overlays. Use fake image `base` with fully saturated color and
    # use real image as alpha channel (transparency)
    base = np.ones_like(images[0])
    for image, color, legend in zip(images, colors, legends):
        base = np.ones_like(image)
        # Custom colormap that contains only `color` at full sauration
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


def plot_spectrum(s, outfile=None):
    """Plots a nice spectrum

        Parameters
        ----------
                s:  ndarray.
                    Spectral data which is expected to cover the energy range
                    0.0 < E <= E_max at an resolution of 0.01 eV per data point.
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

        # Plot full reference spectrum.
        >>>> plot_spectrum(dc.ref_spectrum)

        # Plot and save full reference spectrum.
        >>>> plot_spectrum(dc.ref_spectrum, outfile='ref_spectrum.pdf')
    """
    if outfile:
        ext = os.path.splitext(outfile)[1][1:].lower()
        supported = plt.figure().canvas.get_supported_filetypes()
        assert ext in supported

    x = np.linspace(0, s.shape[0]/100.0, s.shape[0])

    plt.plot(x, s)
    ax = plt.gca()
    ax.set_xlabel('E  [eV]')
    ax.set_ylabel('counts  [-]')

    if outfile:
        plt.savefig(outfile)
