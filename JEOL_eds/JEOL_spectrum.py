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

from datetime import datetime, timedelta
import numpy as np

from JEOL_eds.misc import _decode


class JEOL_spectrum:
    """Reads JEOL spectra ('.eds'.)

    Parameters:
    -----------
    fname : Str
        Filename.

    Examples:
    ---------

    >>> from JEOL_eds import JEOL_spectrum
    >>> import JEOL_eds.utils as JU

    >>> s = JEOL_spectrum('data/spot.eds')

    >>> s.file_name
    'data/spot.eds'

    >>> s.file_date
    '2022-02-17 15:15:20'

    Display meta data, first header:
    >>> _ = s.header
    >>> s.header['username']
    'JEM Administrator'
    >>> s.header['CountRate']
    np.float64(1238.0)

    Now footer:
    >>> _ = s.footer
    >>> s.footer['Parameters']['SEM']
    'JEM-ARM200F(HRP)'
    >>> s.footer['Parameters']['AccKV']
    np.float64(200.0)

    Size of spectral data:
    >>> s.data.shape
    (4096,)

    Plot (uncalibrated) data
    >>> JU.plot_spectrum(s.data,
    ...                  E_range=(0, 20),
    ...                  M_ticks=(4, 1))

    If you need the calibrated data (x-axis)
    >>> import matplotlib.pyplot as plt
    >>> x = range(s.data.shape[0]) * s.header['CoefA'] + s.header['CoefB']
    >>> _ = plt.plot(x, s.data)

    Notes:
    ------
    Reading Analysis Station '.eds' files copied almost verbatim from
    the io_plugin jeol.py of HyperSpy (https://github.com/hyperspy).

    Minor adjustments consist of:
        - slight refactoring of code (__read_eds_header() and
              __read_eds_footer())
        - removing unused variables
        - adjusting meta data structure to JEOL_eds's general layout
        - convertion to a class

    """
    @staticmethod
    def __read_eds_header(fd):
        """Reads the header part of an '.eds' file

        Parameters:
        -----------
        fd : Stream (open file)

        Returns:
        --------
        header : Dict
            Dict with header data (data stored before edx data).
        """
        header = {}
        header["sp_name"] = _decode(fd.read(80).rstrip(b"\x00"))
        header["username"] = _decode(fd.read(32).rstrip(b"\x00"))
        np.fromfile(fd, "<i", 1)  # 1
        header["arr"] = np.fromfile(fd, "<d", 10)
        _ = np.fromfile(fd, "<i", 1)  # 7
        _ = np.fromfile(fd, "<d", 1)[0]  # unknown
        header["Esc"] = np.fromfile(fd, "<d", 1)[0]
        header["Fnano F"] = np.fromfile(fd, "<d", 1)[0]
        header["E Noise"] = np.fromfile(fd, "<d", 1)[0]
        header["CH Res"] = np.fromfile(fd, "<d", 1)[0]
        header["live time"] = np.fromfile(fd, "<d", 1)[0]
        header["real time"] = np.fromfile(fd, "<d", 1)[0]
        header["DeadTime"] = np.fromfile(fd, "<d", 1)[0]
        header["CountRate"] = np.fromfile(fd, "<d", 1)[0]
        header["CountRate n"] = np.fromfile(fd, "<i", 1)[0]
        header["CountRate sum"] = np.fromfile(fd, "<d", 2)
        header["CountRate value"] = np.fromfile(fd, "<d", 1)[0]
        _ = np.fromfile(fd, "<d", 1)[0]  # unknown
        header["DeadTime n"] = np.fromfile(fd, "<i", 1)[0]
        header["DeadTime sum"] = np.fromfile(fd, "<d", 2)
        header["DeadTime value"] = np.fromfile(fd, "<d", 1)[0]
        _ = np.fromfile(fd, "<d", 1)[0]  # unknown
        header["CoefA"] = np.fromfile(fd, "<d", 1)[0]
        header["CoefB"] = np.fromfile(fd, "<d", 1)[0]
        header["State"] = _decode(fd.read(32).rstrip(b"\x00"))
        _ = np.fromfile(fd, "<i", 1)[0]  # unknown
        _ = np.fromfile(fd, "<d", 1)[0]  # unknown
        header["Tpl"] = _decode(fd.read(32).rstrip(b"\x00"))
        header["NumCH"] = np.fromfile(fd, "<i", 1)[0]
        return header

    @staticmethod
    def __read_eds_footer(fd):
        """Reads the footer part of an '.eds' file

        Parameters:
        -----------
        fd : Stream (open file)

        Returns:
        --------
        footer : Dict
        Dict with footer data (data stored after edx data).
        """
        footer = {}
        np.fromfile(fd, "<i", 1)  # unknown

        n_fbd_elem = np.fromfile(fd, "<i", 1)[0]
        if n_fbd_elem != 0:
            list_fbd_elem = np.fromfile(fd, "<H", n_fbd_elem)
            footer["Excluded elements"] = list_fbd_elem

        n_elem = np.fromfile(fd, "<i", 1)[0]
        if n_elem != 0:
            elems = {}
            for _ in range(n_elem):
                _ = np.fromfile(fd, "<i", 1)[0]  # = 2 (mark elem)
                _ = np.fromfile(fd, "<H", 1)[0]  # Z
                _ = np.fromfile(fd, "<i", 1)   # = 1 (mark1)
                _ = np.fromfile(fd, "<i", 1)   # = 0 (mark2)
                roi_min, roi_max = np.fromfile(fd, "<H", 2)
                _ = np.fromfile(fd, "<b", 14)  # unknown
                energy = np.fromfile(fd, "<d", 1)
                _ = np.fromfile(fd, "<d", 1)
                _ = np.fromfile(fd, "<d", 1)
                _ = np.fromfile(fd, "<d", 1)
                elem_name = _decode(fd.read(32).rstrip(b"\x00"))
                _ = np.fromfile(fd, "<i", 1)[0]  # mark3 ?
                n_line = np.fromfile(fd, "<i", 1)[0]
                lines = {}
                for _ in range(n_line):
                    _ = np.fromfile(fd, "<i", 1)[0]  # = 1 (mark line?)
                    e_line = np.fromfile(fd, "<d", 1)[0]
                    z = np.fromfile(fd, "<H", 1)[0]
                    e_length = np.fromfile(fd, "<b", 1)[0]
                    e_name = _decode(fd.read(e_length).rstrip(b"\x00"))
                    l_length = np.fromfile(fd, "<b", 1)[0]
                    l_name = _decode(fd.read(l_length).rstrip(b"\x00"))
                    detect = np.fromfile(fd, "<i", 1)[0]
                    lines[e_name + "_" + l_name] = {
                        "energy": e_line,
                        "Z": z,
                        "detection": detect,
                    }
                elems[elem_name] = {
                    "Z": z,
                    "Roi_min": roi_min,
                    "Roi_max": roi_max,
                    "Energy": energy,
                    "Lines": lines,
                }
            footer["Selected elements"] = elems

        n_quanti = np.fromfile(fd, "<i", 1)[0]
        if n_quanti != 0:
            # all unknown
            _ = np.fromfile(fd, "<i", 1)[0]
            _ = np.fromfile(fd, "<i", 1)[0]
            _ = np.fromfile(fd, "<i", 1)[0]
            _ = np.fromfile(fd, "<d", 1)[0]
            _ = np.fromfile(fd, "<i", 1)[0]
            _ = np.fromfile(fd, "<i", 1)[0]
            quanti = {}
            for _ in range(n_quanti):
                _ = np.fromfile(fd, "<i", 1)[0]  # = 2 (mark elem)
                z = np.fromfile(fd, "<H", 1)[0]
                _ = np.fromfile(fd, "<i", 1)  # = 1 (mark1)
                _ = np.fromfile(fd, "<i", 1)  # = 0 (mark2)
                energy = np.fromfile(fd, "<d", 1)
                _ = np.fromfile(fd, "<d", 1)
                mass1 = np.fromfile(fd, "<d", 1)[0]
                error = np.fromfile(fd, "<d", 1)[0]
                atom = np.fromfile(fd, "<d", 1)[0]
                ox_name = _decode(fd.read(16).rstrip(b"\x00"))
                mass2 = np.fromfile(fd, "<d", 1)[0]
                _ = np.fromfile(fd, "<d", 1)[0]  # K
                counts = np.fromfile(fd, "<d", 1)[0]
                # all unknown
                _ = np.fromfile(fd, "<d", 1)[0]
                _ = np.fromfile(fd, "<d", 1)[0]
                _ = np.fromfile(fd, "<i", 1)[0]
                _ = np.fromfile(fd, "<i", 1)[0]
                _ = np.fromfile(fd, "<d", 1)[0]
                quanti[ox_name] = {
                    "Z": z,
                    "Mass1 (%)": mass1,
                    "Error": error,
                    "Atom (%)": atom,
                    "Mass2 (%)": mass2,
                    "Counts": counts,
                    "Energy": energy,
                }
            footer["Quanti"] = quanti

        e = np.fromfile(fd, "<i", 1)
        if e == 5:
            footer["Parameters"] = {
                "DetT": _decode(fd.read(16).rstrip(b"\x00")),
                "SEM": _decode(fd.read(16).rstrip(b"\x00")),
                "Port": _decode(fd.read(16).rstrip(b"\x00")),
                "AccKV": np.fromfile(fd, "<d", 1)[0],
                "AccNA": np.fromfile(fd, "<d", 1)[0],
                "skip": np.fromfile(fd, "<b", 38),
                "MnKaRES": np.fromfile(fd, "d", 1)[0],
                "WorkD": np.fromfile(fd, "d", 1)[0],
                "InsD": np.fromfile(fd, "d", 1)[0],
                "XtiltAng": np.fromfile(fd, "d", 1)[0],
                "TakeAng": np.fromfile(fd, "d", 1)[0],
                "IncAng": np.fromfile(fd, "d", 1)[0],
                "skip2": np.fromfile(fd, "<i", 1)[0],
                "ScanSize": np.fromfile(fd, "d", 1)[0],
                "DT_64": np.fromfile(fd, "<H", 1)[0],
                "DT_128": np.fromfile(fd, "<H", 1)[0],
                "DT_256": np.fromfile(fd, "<H", 1)[0],
                "DT_512": np.fromfile(fd, "<H", 1)[0],
                "DT_1K": np.fromfile(fd, "<H", 1)[0],
                "DetH": np.fromfile(fd, "d", 1)[0],
                "DirAng": np.fromfile(fd, "d", 1)[0],
                "XtalAng": np.fromfile(fd, "d", 1)[0],
                "ElevAng": np.fromfile(fd, "d", 1)[0],
                "ValidSize": np.fromfile(fd, "d", 1)[0],
                "WinCMat": _decode(fd.read(4).rstrip(b"\x00")),
                "WinCZ": np.fromfile(fd, "<H", 1)[0],
                "WinCThic": np.fromfile(fd, "d", 1)[0],
                "WinChem": _decode(fd.read(16).rstrip(b"\x00")),
                "WinChem_nelem": np.fromfile(fd, "<H", 1)[0],
                "WinChem_Z1": np.fromfile(fd, "<H", 1)[0],
                "WinChem_Z2": np.fromfile(fd, "<H", 1)[0],
                "WinChem_Z3": np.fromfile(fd, "<H", 1)[0],
                "WinChem_Z4": np.fromfile(fd, "<H", 1)[0],
                "WinChem_Z5": np.fromfile(fd, "<H", 1)[0],
                "WinChem_m1": np.fromfile(fd, "d", 1)[0],
                "WinChem_m2": np.fromfile(fd, "d", 1)[0],
                "WinChem_m3": np.fromfile(fd, "d", 1)[0],
                "WinChem_m4": np.fromfile(fd, "d", 1)[0],
                "WinChem_m5": np.fromfile(fd, "d", 1)[0],
                "WinThic": np.fromfile(fd, "d", 1)[0],
                "WinDens": np.fromfile(fd, "d", 1)[0],
                "SpatMat": _decode(fd.read(4).rstrip(b"\x00")),
                "SpatZ": np.fromfile(fd, "<H", 1)[0],
                "SpatThic": np.fromfile(fd, "d", 1)[0],
                "SiDead": np.fromfile(fd, "d", 1)[0],
                "SiThic": np.fromfile(fd, "d", 1)[0],
            }
            return footer

    def __init__(self, fname):
        """Initializes object (reads JEOL spectrum)

        Parameters:
        -----------
        fname : Str
            Filename.
        """
        assert fname.endswith('.eds')
        with open(fname, "br") as fd:
            file_magic = np.fromfile(fd, "<I", 1)[0]
            assert file_magic == 2
            self.file_name = fname
            fd.read(6)  # skip 6 bytes
            self.file_date = (str(datetime(1899, 12, 30) +
                                  timedelta(days=np.fromfile(fd, "<d", 1)[0])))
            self.header = self.__read_eds_header(fd)
            self.data = np.fromfile(fd, "<i", self.header["NumCH"])
            self.footer = self.__read_eds_footer(fd)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
