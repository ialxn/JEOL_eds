#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
Copyright 2020-2021 Ivo Alxneit (ivo.alxneit@psi.ch)

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

from .misc import _decode, _parsejeol



class JEOL_image:
    """Read JEOL image data ('.img' and '.map' files).

        Parameters
        ----------
            fname:      Str
                        Filename.

        Examples
        --------
        >>>> from JEOL_eds import JEOL_image
        >>>> import JEOL_eds.utils as JU

        >>>> demo_im = JEOL_image('data/demo.img')
        >>>> demo_im.file_name
        'data/demo.img'

        >>>> demo_im.file_date
        '2021-08-13 16:09:06'

        # Meta data stored in file.
        >>>> demo_im.parameters
        {'Instrument': {'Type': 0,
          'ScanSize': 198.0,
          'Name': 'JEM-ARM200F(HRP)',
          'AccV': 200.0,
          'Currnnt': 7.475,
          'Mag': 200000,
          'WorkD': 3.2,
          'ScanR': 0.0},
         'FileType': 'JED-2200:IMG',
         'Image': {'Created': 44421.67298611111,
          'GroupName': '',
          'Memo': '',
          'DataType': 1,
          'Size': array([512, 512], dtype=int32),
          'Bits': array([[255, 255, 255, ..., 255, 255, 255],
                 [255, 255, 255, ..., 255, 255, 255],
                 [255, 255, 255, ..., 255, 255, 255],
                 ...,
                 [255, 255, 255, ..., 255, 255, 255],
                 [255, 255, 255, ..., 255, 255, 255],
                 [255, 255, 255, ..., 255, 255, 255]], dtype=uint8),
          'Title': 'IMG1'},
         'Palette': {'RGBQUAD': array([       0,    65793,   131586,   197379,   263172,   328965,
                   394758,   460551,   526344,   592137,   657930,   723723,
                   789516,   855309,   921102,   986895,  1052688,  1118481,
                  ...,
                 16185078, 16250871, 16316664, 16382457, 16448250, 16514043,
                 16579836, 16645629, 16711422, 16777215], dtype=int32),
          '4': {'0': {'Pos': 0, 'Color': 0}, '1': {'Pos': 255, 'Color': 16777215}},
          'Active': 1,
          'Min': 0.0,
          'Max': 255.0,
          'Contrast': 1.0,
          'Brightness': -0.0,
          'Scheme': 1}}

        # Plot image.
        >>>> import matplotlib.pyplot as plt
        >>>> JU.plot_map(demo_im.image, 'Greys_r')

        # Read a map file.
        >>>> demo_map = JEOL_image('data/demo.map')

        # Print calibration data (pixel size in nm).
        >>>> demo_map.nm_per_pixel
        0.99

        # Use ``plot_map()`` for more features
        # ``demo_im``is a BF image. Thus use inverted color map.
        >>>> scale_bar = {'label': '200nm',
                          'f_calib': demo_im.nm_per_pixel,
                          'color': 'white'}
        >>>> JU.plot_map(demo_im.image, 'inferno_r', scale_bar=scale_bar)
    """
    def __init__(self, fname):
        """Initializes object (reads image data).

            Parameters
            ----------
                fname:      Str
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
