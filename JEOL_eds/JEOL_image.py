#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=C0103
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
from datetime import datetime, timedelta
import numpy as np

from JEOL_eds.misc import _decode, _parsejeol



class JEOL_image:
    """Read JEOL image data ('.img' and '.map' files).

    Parameters
    ----------
    fname : Str
        Filename.

    Examples
    --------
    >>> from JEOL_eds import JEOL_image
    >>> import JEOL_eds.utils as JU

    >>> demo_im = JEOL_image('data/demo.img')
    >>> demo_im.file_name
    'data/demo.img'

    >>> demo_im.file_date
    '2021-08-13 16:09:06'

    Meta data stored in file:
    >>> demo_im.parameters['Instrument']['Name']
    'JEM-ARM200F(HRP)'
    >>> demo_im.parameters['Image']['Size']
    array([512, 512], dtype=int32)

    Plot image:
    >>> import matplotlib.pyplot as plt
    >>> JU.plot_map(demo_im.image, 'Greys_r')

    Read a map file:
    >>> demo_map = JEOL_image('data/demo.map')

    Print calibration data (pixel size in nm):
    >>> demo_map.nm_per_pixel
    3.8671875

    Use "plot_map()" for more features. "demo_im" is a BF image thus invert color map:
    >>> scale_bar = {'label': '200nm',
    ...              'f_calib': demo_im.nm_per_pixel,
    ...              'color': 'white'}
    >>> JU.plot_map(demo_im.image, 'inferno_r', scale_bar=scale_bar)
    """
    def __init__(self, fname):
        """Initializes object (reads image data).

        Parameters
        ----------
        fname : Str
            Filename.

        Notes
        -----
        Based on a code fragment by @sempicor at
        https://github.com/hyperspy/hyperspy/pull/2488
        """
        assert os.path.splitext(fname)[1] in ['.img', '.map']
        with open(fname, "br") as fd:
            file_magic = np.fromfile(fd, "<I", 1)[0]
            assert file_magic == 52
            self.file_name = fname
            self.fileformat = _decode(fd.read(32).rstrip(b"\x00"))
            fd.read(8)  # Skip header position and length
            data_pos = np.fromfile(fd, "<I", 1)[0]
            fd.seek(data_pos + 12)
            self.parameters = _parsejeol(fd)
            self.file_date = str(datetime(1899, 12, 30) + timedelta(days=self.parameters["Image"]["Created"]))

            # Make image data easier accessible
            sh = self.parameters["Image"]["Size"]
            self.image = self.parameters["Image"]["Bits"]
            self.image.resize(tuple(sh))

            # Nominal pixel size [nm]
            ScanSize = self.parameters["Instrument"]["ScanSize"]
            Mag = self.parameters["Instrument"]["Mag"]
            self.nm_per_pixel = ScanSize / Mag * 1000000 / sh[0]

if __name__ == "__main__":
    import doctest
    doctest.testmod()
