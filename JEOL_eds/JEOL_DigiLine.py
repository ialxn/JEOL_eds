#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pylint: disable=C0103,C0301
"""
Copyright 2020-2022 Ivo Alxneit (ivo.alxneit@psi.ch)

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
from collections.abc import Iterable
import numpy as np

from JEOL_eds.misc import _parsejeol, _correct_spectrum


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
    0.1546875

    EDX data corresponds to which scan line?
    >>> dl.scan_line
    144

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

        AimArea = self.parameters['EDS Data']['AnalyzableMap MeasData']['Meas Cond']['Aim Area']
        if AimArea[1] != AimArea[3]:
            raise Exception(f'"{fname}" does not contain scan line data! Aim area is {AimArea}')
        # Easy access to scan line index
        self.scan_line = AimArea[1]

        self.dcube = self.__get_data_cube(data_offset)

        # Set MAG calibration factor
        ScanSize = self.parameters['PTTD Param']['Params']['PARAMPAGE0_SEM']['ScanSize']
        Mag = self.parameters['PTTD Data']['AnalyzableMap MeasData']['MeasCond']['Mag']
        self.nm_per_pixel = ScanSize / Mag * 1000000 / self.dcube.shape[1]

        # Make reference spectrum (sum spectrum) accessible easier
        self.ref_spectrum = self.parameters['EDS Data'] \
                                           ['AnalyzableMap MeasData']['Data'] \
                                           ['EDXRF']

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

    def sum_spectrum(self, xRange=None, scans=None):
        """Return sum spectrum for (a fraction of) the scan line (DigiLine).

        Parameters
        ----------
        xRange : Tuple (int, int)
            Section of the scan line (xStart, xStop) to be integrated. Default
            [None] implied the complete scan line.
        scans : Iterable
            Only use scans given for integration.

        Returns
        -------
        spectrum : Ndarray

        Examples
        --------
        >>> from JEOL_eds import JEOL_DigiLine

        Read data:
        >>> dl = JEOL_DigiLine('data/DigiLine/View000_0000003.pts')
        >>> N_s = dl.dcube.shape[0]
        >>> N_s
        50

        Sum spectrumm of odd scans integrated in the range 123-234 (pixels):
        >>> scans = range(1, N_s, 2)
        >>> spectrum = dl.sum_spectrum(scans=scans, xRange=(123, 234))
        >>> spectrum.shape
        (4000,)

        >>> spectrum.sum()
        7259

        Sum spectrum:
        >>> spectrum = dl.sum_spectrum()
        >>> spectrum.sum()
        30710

        This should be close to the reference spectrum:
        >>> ref_spectrum = dl.ref_spectrum
        >>> ref_spectrum.sum()
        30759
        """

        if xRange is None:
            xRange = (0, self.dcube.shape[1])

        assert isinstance(xRange, tuple)
        assert len(xRange) == 2

        if scans is None:
            spectrum = self.dcube[:, xRange[0]:xRange[1], :].sum(axis=(0,1))
        else:
            assert isinstance(scans, Iterable)
            spectrum = np.zeros(self.dcube.shape[-1], dtype='uint32')
            for scan in scans:
                spectrum += self.dcube[scan, xRange[0]:xRange[1], :].sum(axis=0)
        return _correct_spectrum(self.parameters, spectrum)

    def profile(self, interval=None, energy=False, scans=None, xCalib=False):
        """Returns line profile (integrated intensity along scan line).

        Parameters
        ----------
        interval : Tuple (number, number)
            Defines spectral interval (channels, or energy [keV]) to be used
            for profile. None implies that the complete spectrum is integrated.
        energy : Bool
            If False (default) spectral interval is specified as channels
            otherwise (True) interval is specified as energy [keV].
        scans :  Iterable
            Scans included in profile (or None if all scans are used).
        xCalib : Bool
            If set to True x-axis data points are returned as [nm] otherwise
            as pixels.

        Returns
        -------
        x, profile : Ndarray, Ndarray
            X-axis data points (pixel or [nm]) and profile of intergrated
            intensity in interval.

        Examples
        --------
        >>> from JEOL_eds import JEOL_DigiLine

        Read data:
        >>> dl = JEOL_DigiLine('data/DigiLine/View000_0000003.pts')

        Extract Oxygen profile with x axis in [nm]:
        >>> x, p_O = dl.profile(interval=(0.45, 0.6),
        ...                     energy=True, xCalib=True)

        >>> x[0]
        0.0

        >>> x[-1]
        39.4453125

        >>> p_O[0]
        1

        >>> p_O[-1]
        19
        """
        if not interval:
            interval = (0, self.dcube.shape[2])

        if energy:
            CoefA = self.parameters['PTTD Data'] \
                                   ['AnalyzableMap MeasData']['Doc'] \
                                   ['CoefA']
            CoefB = self.parameters['PTTD Data'] \
                                   ['AnalyzableMap MeasData']['Doc'] \
                                   ['CoefB']
            interval = (int(round((interval[0] - CoefB) / CoefA)),
                        int(round((interval[1] - CoefB) / CoefA)))

        if interval[0] > interval[1]:   # ensure interval is (low, high)
            interval = (interval[1], interval[0])

        x = np.arange(float(self.dcube.shape[1]))
        if xCalib:
            x *= self.nm_per_pixel

        if scans is None:
            profile = self.dcube[:, :, interval[0]:interval[1]].sum(axis=(0, 2))
        else:
            profile = np.zeros((self.dcube.shape[1],))
            for scan in scans:
                profile +=self.dcube[scan, :, interval[0]:interval[1]].sum(axis=(1))
        return x, profile

    def spectral_map(self, E_range=None, energy=False):
        """Returns map (spectrum as function of position).

        Parameters
        ----------
        E_range : Tuple (number, number)
            Limit spectral interval (channels, or energy [keV]). None implies
            that the complete spectrum is used.
        energy : Bool
            If False (default) spectral interval is specified as channels
            otherwise (True) interval is specified as energy [keV].

        Returns
        -------
        m : Ndarray
            Spectral map (Energy x Scan).

        Notes
        -----
        The energy axis of the spectral map is always in energy channels. Thus,
        in the example below, data for the first 250 channels (0 < 2.5 keV) is
        returned.

        Examples
        --------
        >>> from JEOL_eds import JEOL_DigiLine

        Read data:
        >>> dl = JEOL_DigiLine('data/DigiLine/View000_0000003.pts')

        Spectral map of Energies up to 2.5 keV:
        >>> m = dl.spectral_map(E_range=(0, 2.5), energy=True)
        >>> m.shape
        (256, 250)
        """
        if not E_range:
            E_range = (0, self.dcube.shape[2])

        if energy:
            CoefA = self.parameters['PTTD Data'] \
                                   ['AnalyzableMap MeasData']['Doc'] \
                                   ['CoefA']
            CoefB = self.parameters['PTTD Data'] \
                                   ['AnalyzableMap MeasData']['Doc'] \
                                   ['CoefB']
            E_range = (int(round((E_range[0] - CoefB) / CoefA)),
                        int(round((E_range[1] - CoefB) / CoefA)))

        if E_range[0] > E_range[1]:   # ensure interval is (low, high)
            E_range = (E_range[1], E_range[0])

        return self.dcube.sum(axis=0)[:, E_range[0]:E_range[1]]

if __name__ == "__main__":
    import doctest
    doctest.testmod()
