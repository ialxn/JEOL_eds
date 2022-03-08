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
import copy
import numpy as np
import matplotlib.pyplot as plt

from JEOL_eds.JEOL_image import JEOL_image
from JEOL_eds.JEOL_spectrum import JEOL_spectrum

from JEOL_eds.misc import _decode



class JEOL_PointLine:
    """Work with JEOL PointLine data (sequence of individual point spectra)

    Parameters
    ----------
    fname : Str
        Filename.

    Examples
    --------

    >>> from JEOL_eds import JEOL_PointLine
    >>> import JEOL_eds.utils as JU

    >>> pl = JEOL_PointLine('data/PointLine/View000_0000001.pln')

    Report some info. '.pln' file contains list of spectra and image:
    >>> pl.file_name
    'View000_0000001.pln'

    >>> pl.Image_name
    'View000_0000000.img'

     ``JEOL_PointLine.eds_dict`` is a dict with marker as key and a list
     [FileName, xPos, yPos] as content:
    >>> pl.eds_dict
    {0: ['View000_0000006.eds', 85.3125, 96.4375], 1: ['View000_0000005.eds', 81.4375, 92.6875], 2: ['View000_0000004.eds', 77.5625, 88.9375], 3: ['View000_0000003.eds', 73.6875, 85.1875], 4: ['View000_0000002.eds', 69.8125, 81.4375]}

    Image object (``JEOL_image``) is stored as ``JEOL_PointLine.ref_image``:
    >>> ref = pl.ref_image

    >>> ref.file_name
    'data/PointLine/View000_0000000.img'

    >>> ref.file_date
    '2022-02-17 15:21:48'

    Image parameters can be accessed such as MAG calibration and image size:
    >>> ref.nm_per_pixel
    1.93359375

    >>> ref.parameters['Image']['Size']
    array([256, 256], dtype=int32)

    The image itself is available too:
    >>> JU.plot_map(ref.image, 'inferno_r')

    Meta data of the first file in ``JEOL_PointList.eds_list``. Most important
    values should be ``CoefA``, ``CoefB`` (calibration of energy axis):
    >>> h = pl.eds_header
    >>> h['CoefA']
    0.0100006
    >>> h['CoefB']
    -0.00122558

    Spectral data is availabe:
    >>> pl.eds_data.shape
    (5, 4096)

    Plot the spectrum corresponding to marker '1':
    >>> JU.plot_spectrum(pl.eds_data[1])
    """
    def __init__(self, fname):
        """Initializes object

        Parameters
        ----------
        fname : Str
            Filename.
        """
        def read_string(fp):
            """Reads string

            Parameter:
            ----------
            fp : File pointer
            """
            assert fp.read(1) == b'\xff'
            fp.read(1)
            str_len = np.fromfile(fp, "<I", 1)[0]
            return _decode(fp.read(str_len).rstrip(b'\x00'))

        def skip_zeros(fp):
            """Skips over a series of b'\x00' bytes

            Parameter:
            ----------
            fp : File pointer
            """
            tmp = fp.read(1)
            while tmp == b'\x00':
                tmp = fp.read(1)

        def find_next_tag(fp, char):
            """Advances File pointer until ``char`` is found (byte)

            Parameter:
            ----------
            fp : File pointer
            char : Byte
            """

            tmp = fp.read(1)
            while tmp != char:
                tmp = fp.read(1)

        def read_eds_meta(fp):
            """Reads meta data bloch for each ".eds" file

            Parameter:
            ----------
            fp : File pointer

            Returns:
            --------
            marker : Int
            [FileName, xPos, yPos] : List
            """
            assert fp.read(1) == b'\xff'
            fp.read(1)
            bytes_len = np.fromfile(fp, "<I", 1)[0]
            Point_nr = int(_decode(fp.read(bytes_len).rstrip(b'\x00')))
            skip_zeros(fp)
            fp.read(4)
            Pos_MM = _decode(fp.read(12).rstrip(b'\x00'))
            find_next_tag(fp, b'\xff')
            fp.read(5)
            Pos_PXL = _decode(fp.read(12).rstrip(b'\x00'))
            fp.read(8)  # skip 2x b'\x08'
            xPos = np.fromfile(fp, "<i", 1)[0]
            yPos = np.fromfile(fp, "<i", 1)[0]
            find_next_tag(fp, b'\xff')
            fp.read(1)
            str_len = np.fromfile(fp, "<I", 1)[0]
            FileName = _decode(fp.read(str_len).rstrip(b'\x00'))
            fp.read(4)
            str_len = np.fromfile(fp, "<I", 1)[0]
            fname = _decode(fp.read(str_len).rstrip(b'\x00'))
            return Point_nr, [fname.rsplit('\\', 1)[1], xPos, yPos]

        # Basic sanity checks
        assert fname.endswith('.pln')
        path, self.file_name = os.path.split(fname)

        # Parse ".pln" file
        with open(fname, 'rb') as fp:
            file_magic = np.fromfile(fp, "<I", 1)[0]    # 10
            assert file_magic == 10
            PointLine = fp.read(9)     # 'PointLine'
            assert PointLine == b'PointLine'
            fp.read(9)  # skip
            fp.read(1)  # 1
            str_len = np.fromfile(fp, "<I", 1)[0]
            ID = _decode(fp.read(str_len).rstrip(b'\x00'))
            fp.read(12)
            Memo = read_string(fp)  # 'Memo'
            fp.read(12)     # skip
            Num = read_string(fp)   # 'Num'
            fp.read(12) # skip
            Image = read_string(fp)     # 'Image'
            fp.read(4)
            str_len = np.fromfile(fp, "<I", 1)[0]
            fn = _decode(fp.read(str_len).rstrip(b'\x00'))
            self.Image_name = fn.rsplit('\\', 1)[1]
            Marker = read_string(fp)    # 'Marker'
            fp.read(10)

            end = False
            self.eds_dict = {}
            while not end:
                try:
                    key, val = read_eds_meta(fp)
                    self.eds_dict[key] = val
                    fp.read(1)
                except:
                    end = True

        # Read and insert reference image
        self.ref_image = JEOL_image(os.path.join(path, self.Image_name))

        # xPos and yPos are in range 0..4096. Rescale to image size.
        for key in self.eds_dict:
            self.eds_dict[key][1] *= (self.ref_image.image.shape[0] / 4096)
            self.eds_dict[key][2] *= (self.ref_image.image.shape[1] / 4096)

        # Read and insert spectral data
        first = True
        for key in self.eds_dict:
            name = self.eds_dict[key][0]
            s = JEOL_spectrum(os.path.join(path, name))
            if first:  # First spectrum read. Perform some initializations
                self.eds_header = copy.deepcopy(s.header)
                self.eds_data = np.zeros((len(self.eds_dict), self.eds_header['NumCH']))
                first = False
            self.eds_data[key] = s.data

    def profile(self, interval=None, energy=False, markers=None):
        """Returns profile of x-ray intensity integrated in `interval`.

        Parameters
        ----------
        interval : Tuple (number, number) or None
            Defines interval (channels, or energy [keV]) to be used for profile.
            None implies that all channels are integrated.
        energy : Bool
            If false (default) interval is specified as channel numbers otherwise
            (True) interval is specified as 'keV'.
        markers : Tuple, List
            Spectra (defined by their markers) included in the profile (or None
            if all spectra are used). The integrated number of counts is set to
            'NaN' for spectra omitted.

        Returns
        -------
        profile : Ndarray
            Profile of intergrated intensity in interval.

        Examples
        --------
        >>> from JEOL_eds import JEOL_PointLine
        >>> import JEOL_eds.utils as JU

        >>> pl = JEOL_PointLine('data/PointLine/View000_0000001.pln')

        Profile of total x-ray intensity:
        >>> p_tot = pl.profile()

        Profile of Ti Ka line with one spectrum (marker '2') omitted.
        >>> p_Ti = pl.profile(interval=(4.4, 4.65),
        ...                   energy=True,
        ...                   markers=[0, 1, 3, 4])

        """
        if not interval:
            interval = (0, self.eds_data.shape[1])

        if energy:
            # Convert to channel numbers
            CoefA = self.eds_header['CoefA']
            CoefB = self.eds_header['CoefB']
            interval = (int(round((interval[0] - CoefB) / CoefA)),
                        int(round((interval[1] - CoefB) / CoefA)))

        if interval[0] > interval[1]:   # ensure interval is (low, high)
            interval = (interval[1], interval[0])

        if markers is None:
            # For consistency, explicitly set dtype to 'float'. We need to
            # allow for NaN in unspecified spectra in the else-clause below.
            profile = self.eds_data[:, interval[0]:interval[1]].sum(axis=1).astype('float')
        else:
            profile = np.full((self.eds_data.shape[0],), np.nan)
            for m in markers:
                profile[m] =self.eds_data[m, interval[0]:interval[1]].sum()
        return profile

    def show_PointLine(self, ROI=None,
                       color='white', ann_color='black',
                       outfile=None):
        """Plots definition (points / markers) of Pointline on reference image.

        Parameters
        ----------
        ROI : Tuple (top, bottom, left, right), 'auto', or [None]
            Specific region to zoom in or use automatically chosen region.
            Default [None] is use full image.
        color : Str
            Color used to draw markers (plus sign) ['white'].
        ann_color : Str
        Color used to draw marker annotations ['black'].
        outfile : Str
            Filename (optional) to store plot.

        Examples
        --------
        >>> from JEOL_eds import JEOL_PointLine
        >>> pl = JEOL_PointLine('data/PointLine/View000_0000001.pln')
        >>> pl.show_PointLine(ROI=(45,110,50,100),
        ...                   color='red',
        ...                   ann_color='blue')
        """
        # Reference image
        plt.imshow(self.ref_image.image)

        # Plot '+' and annotate at PointLine positions
        for key in self.eds_dict:
            xPos = self.eds_dict[key][1]
            yPos = self.eds_dict[key][2]
            plt.plot(xPos, yPos, '+', color=color)
            plt.annotate(key, (xPos, yPos), color=ann_color)

        # Use xlim, ylim for zoomed in region.
        # y axis in image is inverted (origin is top/left). Thus
        # inverte order of ylim coordinates.
        if not ROI:
            xlim = (0, self.ref_image.image.shape[0])
            ylim = (self.ref_image.image.shape[1], 0)
        elif ROI == 'auto':
            xmin = np.asarray([self.eds_dict[k][1] for k in self.eds_dict]).min()
            xmax = np.asarray([self.eds_dict[k][1] for k in self.eds_dict]).max()
            dx = (xmax - xmin) / 10.0
            xlim = (xmin - dx, xmax + dx)
            ymin = np.asarray([self.eds_dict[k][2] for k in self.eds_dict]).min()
            ymax = np.asarray([self.eds_dict[k][2] for k in self.eds_dict]).max()
            dy = (ymax - ymin) / 10.0
            ylim = (ymin - dy, ymax + dy)
        else:
            xlim = (ROI[2], ROI[3])
            ylim = (ROI[1], ROI[0])
        plt.xlim(xlim)
        plt.ylim(ylim)

        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])

        if outfile:
            plt.savefig(outfile)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
