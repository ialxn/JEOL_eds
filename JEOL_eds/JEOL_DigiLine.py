#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pylint: disable=C0103,C0301
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
from datetime import datetime, timedelta
import numpy as np

from JEOL_eds.misc import _parsejeol


class JEOL_DigiLine:
    """Work with Analysis Station DigiLine data (spectra collected for a single scan line).

    Parameters
    ----------
    fname : Str
        Filename.

    Notes
    -----
        Much of this class is an identical or simplified version of
        ``JEOL_eds.JEOL_pts()``. For implementation details see comments in
        this class.

    Examples
    --------
    >>> from JEOL_eds import JEOL_DigiLine

    Read data:
    >>> dl = JEOL_DigiLine('data/DigiLine/View000_0000003.pts')

    Report some meta data.
    >>> dl.file_name
    'data/DigiLine/View000_0000003.pts'

    >>> dl.file_date
    '2022-03-04 13:19:43'

    Mag calibration factor:
    >>> dl.nm_per_pixel
    0.0099

    Full parameter set stored by Analysis Station is available via the attribute
    ``parameters``. Here we query LiveTime:
    >>> dl.parameters['PTTD Data']['AnalyzableMap MeasData']['Doc']['LiveTime']
    63.13

    Data cube N x X x E (N_scans x N_pixels x N_E-channels
    >>> dl.dcube.shape
    (50, 256, 4000)
    """
    def __init__(self, fname):
        assert fname.endswith('pts')
        self.file_name = fname
        self.parameters, data_offset = self.__parse_header(fname)
        self.dcube = self.__get_data_cube(data_offset)

        # Set MAG calibration factor
        ScanSize = self.parameters['PTTD Param']['Params']['PARAMPAGE0_SEM']['ScanSize']
        Mag = self.parameters['PTTD Data']['AnalyzableMap MeasData']['MeasCond']['Mag']
        self.nm_per_pixel = ScanSize / Mag * 1000000 / self.dcube.shape[2]


    def __parse_header(self, fname):
        """Extract meta data from header in JEOL ".pts" file.

        Parameters
        ----------
        fnamec : Str
            Filename.

        Returns
        -------
        header : Dict
            Dictionary containing all meta data stored in header.
        offset : Int

        Notes
        -----
            Copied almost verbatim from Hyperspy (hyperspy/io_plugins/jeol.py).
        """
        with open(fname, 'br') as fd:
            file_magic = np.fromfile(fd, '<I', 1)[0]
            assert file_magic == 304
            fd.read(16)
            head_pos = np.fromfile(fd, '<I', 1)[0]
            fd.read(4)  # Skip header length
            data_pos = np.fromfile(fd, '<I', 1)[0]
            fd.read(264)
            self.file_date = (str(datetime(1899, 12, 30) +
                                  timedelta(days=np.fromfile(fd, 'd', 1)[0])))
            fd.seek(head_pos + 12)
            return _parsejeol(fd), data_pos


    def __CH_offset_from_meta(self):
        """Returns offset (channel corresponding to E=0).
        """
        Tpl_cond = self.parameters['EDS Data'] \
                                  ['AnalyzableMap MeasData']['Meas Cond'] \
                                  ['Tpl']
        return self.parameters['PTTD Param'] \
                              ['Params']['PARAMPAGE1_EDXRF']['Tpl'][Tpl_cond] \
                              ['DigZ']

    def __get_data_cube(self, offset):
        """Returns data cube (N x X x E).

        Parameters
        ----------
        offset : Int
            Number of header bytes.

        Returns
        -------
        dcube : Ndarray (N x X x numCH)
            Data cube. N is the number of sweeps, X the length of the scan
            (width of image), and numCH the number of energy channels of the
            spectra.

        Notes
        -----
            This is a much simplified version of ``JEOL_pts.__get_data_cube()``.
            For details see there.
        """
        CH_offset = self.__CH_offset_from_meta()
        NumCH = self.parameters['PTTD Param'] \
                               ['Params']['PARAMPAGE1_EDXRF'] \
                               ['NumCH']
        N_spec = NumCH - CH_offset

        AimArea = self.parameters['EDS Data'] \
                                 ['AnalyzableMap MeasData']['Meas Cond'] \
                                 ['Aim Area']
        assert AimArea[1] == AimArea[3] # This is a scan line

        Sweep = self.parameters['PTTD Data'] \
                               ['AnalyzableMap MeasData']['Doc'] \
                               ['Sweep']
        dcube = np.zeros([Sweep, AimArea[2] + 1, N_spec],
                         dtype='uint32')

        # Read data into buffer
        with open(self.file_name, 'rb') as f:
            f.seek(offset)
            data = np.fromfile(f, dtype='u2')

        N = 0
        scan = 0
        x = -1
        END = 45056 + NumCH
        scale = 4096 / (AimArea[2] + 1)

        for d in data:
            N += 1
            if 32768 <= d < 36864:
                d = int((d - 32768) / scale)
                if d < x:
                    # A new scan starts
                    scan += 1
                x = d
            elif 36864 <= d < 40960:    # Will be ScanLine index (uninteresting because is constant)
                pass
            elif 45056 <= d < END:
                E = int(d - 45056)
                E -= CH_offset
                dcube[scan, x, E] = dcube[scan, x, E] + 1
            else:   # Unknown data
                pass
        return dcube

if __name__ == "__main__":
    import doctest
    doctest.testmod()
