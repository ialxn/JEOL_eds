#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 15:11:53 2021

@author: alxneit
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def create_overlay(images, colors):
    """Plots overlay of `images` with `colors`.

        Parameters
        ----------
           images:  List or tuple.
                    Images to be overlayed (must be of identical shape).
           colors:  List or tuple.
                    List of colors used to overlay the images provided
                    in the same order as `images`. Surplus colors provided
                    are ignored.

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
        # iron. Focus is on iron distribution.
        >>>> create_overlay((O, Si, Fe), ('Red', 'Green', 'Blue'))

        # Focus on oxygen. Follows both, iron and silicon distributions.
        >>>> create_overlay([Fe, Si, O], ['Blue', 'Red', 'Green'])
    """
    assert isinstance(images, (list, tuple))
    assert isinstance(colors, (list, tuple))
    assert len(colors) >= len(images)   # Sufficient colors provided
    assert all(image.shape==images[0].shape for image in images)    # all same shape

    # Create overlays. Use fake image `base` with fully saturated color and
    # use real image as alpha channel (transparency)
    base = np.ones_like(images[0])
    for image, color in zip(images, colors):
        # Custom colormap that contains only `color` at full sauration
        cmap = LinearSegmentedColormap.from_list("cmap", (color, color))
        alpha = image / image.max()
        plt.imshow(base, cmap=cmap, alpha=alpha,
                   vmin=0, vmax=1)
