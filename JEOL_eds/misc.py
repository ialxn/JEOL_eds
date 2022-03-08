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

import numpy as np

def _decode(bytes_string):
    try:
        string = bytes_string.decode("utf-8")
    except:
        # See https://github.com/hyperspy/hyperspy/issues/2812
        string = bytes_string.decode("shift_jis")

    return string

def _parsejeol(fd):
    """Parse meta data.

        Parameters
        ----------
               fd:     File descriptor positioned at start of parseable header

           Returns
           -------
                       Dict
                       Dictionary containing all meta data stored in header.

             Notes
             -----
                       Copied almost verbatim from Hyperspy (hyperspy/io_plugins/jeol.py).
    """
    jTYPE = {
        1: 'B',
        2: 'H',
        3: 'i',
        4: 'f',
        5: 'd',
        6: 'B',
        7: 'H',
        8: 'i',
        9: 'f',
        10: 'd',
        11: '?',
        12: 'c',
        13: 'c',
        14: 'H',
        20: 'c',
        65553: '?',
        65552: '?',
        }

    final_dict = {}
    tmp_list = []
    tmp_dict = final_dict
    mark = 1
    while abs(mark) == 1:
        mark = np.fromfile(fd, 'b', 1)[0]
        if mark == 1:
            str_len = np.fromfile(fd, '<i', 1)[0]
            kwrd = fd.read(str_len).rstrip(b'\x00')
            if (
                kwrd == b'\xce\xdf\xb0\xc4'
            ):  # correct variable name which might be 'Port'
                kwrd = 'Port'
            elif (
                kwrd[-1] == 222
            ):  # remove undecodable byte at the end of first ScanSize variable
                kwrd = kwrd[:-1].decode('utf-8')
            else:
                kwrd = kwrd.decode('utf-8')
            val_type, val_len = np.fromfile(fd, '<i', 2)
            tmp_list.append(kwrd)
            if val_type == 0:
                tmp_dict[kwrd] = {}
            else:
                c_type = jTYPE[val_type]
                arr_len = val_len // np.dtype(c_type).itemsize
                if c_type == 'c':
                    value = fd.read(val_len).rstrip(b'\x00')
                    value = value.decode('utf-8').split('\x00')
                    # value = os.path.normpath(value.replace('\\','/')).split('\x00')
                else:
                    value = np.fromfile(fd, c_type, arr_len)
                if len(value) == 1:
                    value = value[0]
                if kwrd[-5:-1] == 'PAGE':
                    kwrd = kwrd + '_' + value
                    tmp_dict[kwrd] = {}
                    tmp_list[-1] = kwrd
                elif kwrd in ('CountRate', 'DeadTime'):
                    tmp_dict[kwrd] = {}
                    tmp_dict[kwrd]['value'] = value
                elif kwrd == 'Limits':
                    pass
                        # see https://github.com/hyperspy/hyperspy/pull/2488
                        # first 16 bytes are encode in float32 and looks like
                        # limit values ([20. , 1., 2000, 1.] or [1., 0., 1000., 0.001])
                        # next 4 bytes are ASCII character and looks like
                        # number format (%.0f or %.3f)
                        # next 12 bytes are unclear
                        # next 4 bytes are ASCII character and are units (kV or nA)
                        # last 12 byes are unclear
                elif val_type == 14:
                    tmp_dict[kwrd] = {}
                    tmp_dict[kwrd]['index'] = value
                else:
                    tmp_dict[kwrd] = value
            if kwrd == 'Limits':
                pass
                    # see https://github.com/hyperspy/hyperspy/pull/2488
                    # first 16 bytes are encode in int32 and looks like
                    # limit values (10, 1, 100000000, 1)
                    # next 4 bytes are ASCII character and looks like number
                    # format (%d)
                    # next 12 bytes are unclear
                    # next 4 bytes are ASCII character and are units (mag)
                    # last 12 byes are again unclear
            else:
                tmp_dict = tmp_dict[kwrd]
        else:
            if len(tmp_list) != 0:
                del tmp_list[-1]
                tmp_dict = final_dict
                for k in tmp_list:
                    tmp_dict = tmp_dict[k]
            else:
                mark = 0
    return final_dict

def _correct_spectrum(parameters, s):
    """Apply non-linear energy correction at low energies to spectrum.

    Parameters
    ----------
    parameters : dict
        Complete dict of meta data.
    s : Ndarray
        Uncorrected spectrum.

    Returns
    -------
    s : Ndarray
    Original or corrected spectrum, depending on whether correction is necessary.
    ~"""
    def apply_correction(s, ExCoef):
        """Applies the correction formula.

        Parameters
        ----------
        s :  Ndarray
            Original spectrum
        ExCoef :  List
            Correction coefficients.

        Returns
        -------
        s : Ndarray
            Corrected spectrum.
        """
        CH_Res = parameters['PTTD Param'] \
                           ['Params']['PARAMPAGE1_EDXRF'] \
                           ['CH Res']
        E_uncorr = np.arange(0, ExCoef[3], CH_Res)
        N = E_uncorr.shape[0]
        ###########################################################
        #                                                         #
        # Correction formula (guess) using the three parameters   #
        # given in `ExCoef`.                                      #
        #                                                         #
        # The correction does not yet yield exactly the reference #
        # spectrum at EDXRF. Peak positions are matched well but  #
        # the line shape still shows some differences. I guess    #
        # that this is related to the interpolation part.         #
        #                                                         #
        # With 'data/128.pts' as example:                         #
        #     >>> ref_spec[0:100].sum()                          #
        #     200468                                              #
        #     >>> corrected_spec[0:100].sum()                    #
        #     200290                                              #
        #                                                         #
        ###########################################################
        E_corr = ExCoef[0]*E_uncorr**2 + ExCoef[1]*E_uncorr + ExCoef[2]
        s[0:N] = np.interp(E_uncorr, E_corr, s[0:N])
        return s

    Tpl_cond = parameters['EDS Data'] \
                         ['AnalyzableMap MeasData']['Meas Cond'] \
                         ['Tpl']
    try:
        ExCoef = parameters['PTTD Param'] \
                           ['Params']['PARAMPAGE1_EDXRF']['Tpl'][Tpl_cond] \
                           ['ExCoef']
        return apply_correction(s, ExCoef)
    except KeyError:
        return s