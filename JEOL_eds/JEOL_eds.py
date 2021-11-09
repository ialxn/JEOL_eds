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
import os
import sys
from datetime import datetime, timedelta
from warnings import warn
import h5py
import asteval
import numpy as np
from scipy.signal import wiener, correlate
import matplotlib.pyplot as plt
import matplotlib.animation as animation



def decode(bytes_string):
    try:
        string = bytes_string.decode("utf-8")
    except:
        # See https://github.com/hyperspy/hyperspy/issues/2812
        string = bytes_string.decode("shift_jis")

    return string

def parsejeol(fd):
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
                        # next 4 bytes are ascii character and looks like
                        # number format (%.0f or %.3f)
                        # next 12 bytes are unclear
                        # next 4 bytes are ascii character and are units (kV or nA)
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
                    # next 4 bytes are ascii character and looks like number
                    # format (%d)
                    # next 12 bytes are unclear
                    # next 4 bytes are ascii character and are units (mag)
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


class JEOL_pts:
    """Work with JEOL '.pts' files

        Parameters
        ----------
            fname:      Str
                        Filename.
            dtype:      Str
                        Data type used to store (not read) data cube.
                        Can be any of the dtype supported by numpy.
                        If a '.npz' file is loaded, this parameter is
                        ignored and the dtype corresponds to the one
                        of the data cube when it was stored.
     split_frames:      Bool
                        Store individual frames in the data cube (if
                        True), otherwise add all frames and store in
                        a single frame (default).
       frame_list:      List (or None)
                        List of frams to be read if split_frames was specified.
                        Default (None) implies all frames present in data are
                        read.
         E_cutoff:      Float
                        Energy cutoff in spectra. Only data below E_cutoff
                        are read.
       read_drift:      Bool
                        Read BF images (one BF image per frame stored in
                        the raw data, if the option "correct for sample
                        movement" was active while the data was collected).
                        All images are read even if only a subset of frames
                        is read (frame_lsit is specified).
    only_metadata:      Bool
                        Only meta data is read (True) but nothing else. All
                        other keywords are ignored.
          verbose:      Bool
                        Turn on (various) output.

        Examples
        --------

        >>>> from JEOL_eds import JEOL_pts

        # Initialize JEOL_pts object (read data from '.pts' file).
        # Data cube has dtype = 'uint16' (default).
        >>>> dc = JEOL_pts('128.pts')
        >>>> dc.dcube.shape     # Shape of data cube
        (1, 128, 128, 4000)
        >>>> dc.dcube.dtype
        dtype('uint16')

        # Same, but specify dtype to be used for data cube.
        >>>> dc = JEOL_pts('128.pts', dtype='int')
        >>>> dc.dcube.dtype
        dtype('int64')

        # Provide additional (debug) output when loading.
        >>>> dc = JEOL_pts('128.pts', dtype='uint16', verbose=True)
        Unidentified data items (2081741 out of 82810, 3.98%) found:
	        24576: found 41858 times
	        28672: found 40952 times

        # Store individual frames.
        >>>> dc=JEOL_pts('128.pts', split_frames=True)
        >>>> dc.dcube.shape
        (50, 128, 128, 4000)

        # For large data sets, read only a subset of frames.
        >>>> small_dc = JEOL_pts('128.pts',
                                 split_frames=True, list_frames=[1,2,4,8,16])
        +>>>> small_dc.frame_list
        [1, 2, 4, 8, 16]
        >>>> small_dc.dcube.shape
        (5, 128, 128, 1100)
        # The frames in the data cube correspond to the original frames 1, 2,
        # 4, 8, and 16.

        # Only import spectrum up to cutoff energy [keV]
        >>>> dc = JEOL_pts('128.pts', E_cutoff=10.0)
        >>>> dc.dcube.shape
        (1, 128, 128, 1000)

        # Also read and store BF images (one per frame) present.
        # Attribute will be set to 'None' if no data was not read
        # or found!
        >>> dc = JEOL_pts('128.pts', read_drift=True)
        dc.drift_images is None
        False
        >>> dc.drift_images.shape
        (50, 128, 128)
        # If only a subset of frames was read select the corresponding
        # BF images as follows:
        >>>> [dc.drift_images[i] for i in dc.frame_list]
        [array([[67, 70, 57, ..., 37, 39, 41],
                [68, 70, 63, ..., 43, 44, 39],
                [68, 67, 58, ..., 48, 47, 47],
                ...,
                [64, 63, 61, ..., 58, 66, 68],
                [59, 53, 57, ..., 60, 58, 58],
                [56, 67, 68, ..., 62, 56, 58]], dtype=uint16)]

        # Useful attributes.
        >>>> dc.file_name
        '128.pts'               # File name loaded from.
        >>>> dc.file_date
        '2020-10-23 11:18:40'   # File creation date

        # More info is stored in attribute `JEOL_pts.parameters`.
        >>>> dc.parameters      # Full dict
        {'PTTD Cond': {'Meas Cond': {'CONDPAGE0_C.R': {'Tpl': {'index': 3,
             'List': ['T1', 'T2', 'T3', 'T4']},
        .
        .
        .
            'FocusMP': 16043213}}}}

        # Measurement parameters active when map was acquired.
        >>>> dc.parameters['EDS Data']['AnalyzableMap MeasData']['Doc']
        {'LiveTime': 409.5,
         'RealTime': 418.56,
         'CountRate': {'value': 538,
          'n': 1085,
          'sum': array([7.84546000e+05, 6.31998988e+08])},
         'DeadTime': {'value': 1, 'n': 1085, 'sum': array([2025., 4409.])},
         'DwellTime(msec)': 0.5,
         'Sweep': 50,
         'ScanLine': 128,
         'CoefA': 0.0100006,
         'CoefB': -0.00122558,
         'Esc': 1.75}

        # JEOL_pts objects can also be initialized from a saved data cube. In
        # this case, the dtype of the data cube is the same as in the stored
        # data and a possible 'dtype=' keyword is ignored.
        # This only initializes the data cube. Most attributes are not loaded
        # and are set to 'None'
        >>>> dc2 = JEOL_pts('128.npz')
        >>>> dc2.file_name
        '128.npz'
        >>>> dc2.parameters is None
        True

        # Additionally, JEOL_pts object can be saved as hdf5 files.
        # This has the benefit that all attributes (drift_images, parameters)
        # are also stored.
        # Use base name of original file and pass along keywords to
        # `h5py.create_dataset()`.
        >>>> dc.save_hdf5(compression='gzip', compression_opts=9)

        # Initialize from hdf5 file. Only filename is used, additional keywords
        # are ignored.
        >>>> dc3 = JEOL_pts('128.h5')
        >>>> dc3.parameters
        {'PTTD Cond': {'Meas Cond': {'CONDPAGE0_C.R': {'Tpl': {'index': 3,
             'List': ['T1', 'T2', 'T3', 'T4']},
        .
        .
        .
            'FocusMP': 16043213}}}}

        # Fast way to read and plot reference spectrum.
        >>>> plot_spectrum(JEOL_pts('64.pts', only_metadata=True).ref_spectrum)
    """

    def __init__(self, fname, dtype='uint16',
                 split_frames=False, frame_list=None,
                 E_cutoff=False, read_drift=False,
                 only_metadata=False, verbose=False):
        """Reads data cube from JEOL '.pts' file or from previously saved data cube.

            Parameters
            ----------
                 fname:     Str
                            Filename.
                 dtype:     Str
                            Data type used to store (not read) data cube.
                            Can be any of the dtype supported by numpy.
                            If a '.npz' file is loaded, this parameter is
                            ignored and the dtype corresponds to the one
                            of the data cube when it was stored.
          split_frames:     Bool
                            Store individual frames in the data cube (if
                            True), otherwise add all frames and store in
                            a single frame (default).
            frame_list:     List
                            List of frams to be read if split_frames was
                            specified. Default (None) implies all frames
                            present in data are read.
              E_cutoff:     Float
                            Energy cutoff in spectra. Only data below E_cutoff
                            are read.
            read_drift:     Bool
                            Read BF images (one BF image per frame stored in
                            the raw data, if the option "correct for sample
                            movement" was active while the data was collected).
                            All images are read even if only a subset of frames
                            is read (frame_lsit is specified).
         only_matadata:     Bool
                            Only metadata are read (True) but nothing else. All
                            other keywords are ignored.
               verbose:     Bool
                            Turn on (various) output.
        """
        if os.path.splitext(fname)[1] == '.pts':
            self.file_name = fname
            self.parameters, data_offset = self.__parse_header(fname)
            if only_metadata:
                self.dcube = None
                self.drift_images = None
                self.frame_list = None
                self.__set_ref_spectrum()
                return

            self.frame_list = sorted(list(frame_list)) if split_frames and frame_list else None
            self.drift_images = self.__read_drift_images(fname) if read_drift else None
            self.dcube = self.__get_data_cube(dtype, data_offset,
                                              split_frames=split_frames,
                                              E_cutoff=E_cutoff,
                                              verbose=verbose)

        elif os.path.splitext(fname)[1] == '.npz':
            self.parameters = None
            self.__load_dcube(fname)

        elif os.path.splitext(fname)[1] == '.h5':
            self.__load_hdf5(fname)

        else:
            raise OSError(f"Unknown type of file '{fname}'")

        self.__set_ref_spectrum()


    def __set_ref_spectrum(self):
        """Sets attribute ref_spectrum from parameters dict.
        """
        if self.parameters:
            # Determine length of ref_spectrum.
            try:
                N = self.dcube.shape[3]
            except AttributeError:
                N = self.parameters['PTTD Param']['Params']['PARAMPAGE1_EDXRF']['NumCH']
                # We use 1000, 2000, 4000 channels (no negative energies)
                N = N // 1000 * 1000
            self.ref_spectrum = self.parameters['EDS Data'] \
                                               ['AnalyzableMap MeasData']['Data'] \
                                               ['EDXRF'][0:N]
        else:
            self.ref_spectrum = None


    def __parse_header(self, fname):
        """Extract meta data from header in JEOL ".pts" file.

            Parameters
            ----------
                fname:  Str
                        Filename.

            Returns
            -------
                        Dict
                        Dictionary containing all meta data stored in header.
            Notes
            -----
                    Copied almost verbatim from Hyperspy (hyperspy/io_plugins/jeol.py).
        """
        with open(fname, 'br') as fd:
            file_magic = np.fromfile(fd, '<I', 1)[0]
            assert file_magic == 304
            fd.read(16)
            head_pos, head_len, data_pos, data_len = np.fromfile(fd, '<I', 4)
            fd.read(260)
            self.file_date = (str(datetime(1899, 12, 30) +
                                  timedelta(days=np.fromfile(fd, 'd', 1)[0])))
            fd.seek(head_pos + 12)
            return parsejeol(fd), data_pos


    def __CH_offset_from_meta(self):
        """Returns offset (channel corresponding to E=0).
        """
        Tpl_cond = self.parameters['EDS Data'] \
                                  ['AnalyzableMap MeasData']['Meas Cond'] \
                                  ['Tpl']
        return self.parameters['PTTD Param'] \
                              ['Params']['PARAMPAGE1_EDXRF']['Tpl'][Tpl_cond] \
                              ['DigZ']

    def __get_data_cube(self, dtype, offset, split_frames=False,
                        E_cutoff=None, verbose=False):
        """Returns data cube (F x X x Y x E).

            Parameters
            ----------
                dtype:      Str
                            Data type used to store data cube in memory.
                hsize:      Int
                            Number of header bytes.
         split_frames:      Bool
                            Store individual frames in the data cube (if
                            True), otherwise add all frames and store in
                            a single frame (default).
             E_cutoff:      Float
                            Cutoff energy for spectra. Only store data below
                            this energy.
              verbose:      Bool
                            Print additional output

            Returns
            -------
                dcube:      Ndarray (N x size x size x numCH)
                            Data cube. N is the number of frames (if split_frames
                            was selected) otherwise N=1, image is size x size pixels,
                            spectra contain numCH channels.
        """
        CH_offset = self.__CH_offset_from_meta()
        NumCH = self.parameters['PTTD Param'] \
                               ['Params']['PARAMPAGE1_EDXRF'] \
                               ['NumCH']
        area = self. parameters['EDS Data'] \
                               ['AnalyzableMap MeasData']['Meas Cond'] \
                               ['Pixels'].split('x')
        h = int(area[0])
        v = int(area[1])
        if E_cutoff:
            CoefA = self.parameters['PTTD Data'] \
                                   ['AnalyzableMap MeasData']['Doc'] \
                                   ['CoefA']
            CoefB = self.parameters['PTTD Data'] \
                                   ['AnalyzableMap MeasData']['Doc'] \
                                   ['CoefB']
            N_spec = round((E_cutoff - CoefB) / CoefA)
        else:
            N_spec = NumCH - CH_offset
        with open(self.file_name, 'rb') as f:
            f.seek(offset)
            data = np.fromfile(f, dtype='u2')
        if split_frames:
            Sweep = self.parameters['PTTD Data'] \
                                   ['AnalyzableMap MeasData']['Doc'] \
                                   ['Sweep']
            if self.frame_list:
                # Check that only frames present in data are requested.
                if not all(x < Sweep for x in self.frame_list):
                    # Make list with frames request that ARE present.
                    self.frame_list = [x for x in self.frame_list if x < Sweep]
                # Fewer frames requested than present, update Sweep
                # to allocate smaller dcube.
                Sweep = len(self.frame_list)
            dcube = np.zeros([Sweep, v, h, N_spec],
                             dtype=dtype)
        else:
            dcube = np.zeros([1, v, h, N_spec],
                             dtype=dtype)
        N = 0
        N_err = 0
        unknown = {}
        frame = 0
        x = -1
        # Data is mapped as follows:
        #   32768 <= datum < 36864                  -> y-coordinate
        #   36864 <= datum < 40960                  -> x-coordinate
        #   45056 <= datum < END (=45056 + NumCH)    -> count registered at energy
        END = 45056 + NumCH
        scale = 4096 / h
        # map the size x size image into 4096x4096
        for d in data:
            N += 1
            if 32768 <= d < 36864:
                y = int((d - 32768) / scale)
            elif 36864 <= d < 40960:
                d = int((d - 36864) / scale)
                if split_frames and d < x:
                    # A new frame starts once the slow axis (x) restarts. This
                    # does not necessary happen at x=zero, if we have very few
                    # counts and nothing registers on first scan line.
                    frame += 1
                    try:
                        if frame > max(self.frame_list):
                            # Further frames present are not required, so stop
                            # (slow) reading and return data read (dcube).
                            return dcube
                    except TypeError:
                        pass
                x = d
            elif 45056 <= d < END:
                z = int(d - 45056)
                z -= CH_offset
                if N_spec > z >= 0:
                    try:    # self.frame_list might be None
                        if frame in self.frame_list:
                            # Current frame is specified in self.frame_list.
                            # Store data in self.dcube in correct position
                            # i.e. position within list.
                            idx = self.frame_list.index(frame)
                            dcube[idx, x, y, z] = dcube[idx, x, y, z] + 1
                    except TypeError:
                        # self.frame_list is None, just store data in this frame
                        dcube[frame, x, y, z] = dcube[frame, x, y, z] + 1
            else:
                if verbose:
                    if 40960 <= d < 45056:
                        # Image (one per sweep) stored if option
                        # "correct for sample movement" was active
                        # during data collection.
                        continue
                    if str(d) in unknown:
                        unknown[str(d)] += 1
                    else:
                        unknown[str(d)] = 1
                    N_err += 1
        if verbose:
            print(f'Unidentified data items ({N_err} out of {N}, '
                  f'{100 * N_err / N:.2f}%) found:')
            for key in sorted(unknown):
                print(f'\t{key}: found {unknown[key]}')
        return dcube

    def __read_drift_images(self, fname):
        """Read BF images stored in raw data

            Parameters
            ----------
                fname:      Str
                            Filename.

            Returns
            -------
                Ndarray or None if data is not available
                Stack of images with shape (N_images, im_size, im_size)

        Notes
        -----
            Based on a code fragment by @sempicor at
            https://github.com/hyperspy/hyperspy/pull/2488
        """
        N_images = self.parameters['PTTD Data'] \
                                  ['AnalyzableMap MeasData']['Doc'] \
                                  ['Sweep']
        area = self. parameters['EDS Data'] \
                               ['AnalyzableMap MeasData']['Meas Cond'] \
                               ['Aim Area']
        h = area[2] - area[0] + 1
        v = area[3] - area[1] + 1
        image_shape = (N_images, v, h)
        with open(fname) as f:
            f.seek(28)  # see self.__parse_header()
            data_pos = np.fromfile(f, '<I', 1)[0]
            f.seek(data_pos)
            rawdata = np.fromfile(f, dtype='u2')
            ipos = np.where(np.logical_and(rawdata >= 40960, rawdata < 45056))[0]
            if len(ipos) == 0:  # No data available
                return None
            I = np.array(rawdata[ipos] - 40960, dtype='uint16')
            try:
                return I.reshape(image_shape)
            except ValueError:  # incomplete image
                # Add `N_addl` NaNs before reshape()
                N_addl = N_images * v * h - I.shape[0]
                I = np.append(I, np.full((N_addl), np.nan, dtype='uint16'))
                return I.reshape(image_shape)


    def drift_statistics(self, filtered=False, verbose=False):
        """Returns 2D frequency distribution of frame shifts (x, y).

            Parameters
            ----------
             filtered:     Bool
                           If True, use Wiener filtered data.
              verbose:     Bool
                           Provide additional info if set to True.

           Returns
           -------
                    h:     Ndarray or None if data cube contains a single
                           frame only.
               extent:     List
                           Used to plot histogram as plt.imshow(h, extent=extent)

            Examples
            --------

            # Calculate the 2D frequency distribution of the frames shifts
            # using unfiltered frames (verbose output).
            >>>> dc.drift_statistics(verbose=True)
            Frame 0 used a reference
            Average of (-2, -1) (0, 0) set to (-1, 0) in frame 24
            Shifts (unfiltered):
                Range: -2 - 1
                Maximum 12 at (0, 0)
            (array([[ 0.,  0.,  5.,  0.,  0.],
                    [ 0.,  9.,  7.,  0.,  0.],
                    [ 1., 10., 12.,  1.,  0.],
                    [ 0.,  0.,  4.,  1.,  0.],
                    [ 0.,  0.,  0.,  0.,  0.]]),
             [-2, 2, -2, 2])

            # Return the 2D frequency distribution of the Wiener filtered
            # frames (plus extent useful for plotting).
            >>>> m, e = dc.drift_statistics(filtered=True)
            /.../scipy/signal/signaltools.py:1475: RuntimeWarning: divide by zero encountered in true_divide
              res *= (1 - noise / lVar)
            /.../scipy/signal/signaltools.py:1475: RuntimeWarning: invalid value encountered in multiply
              res *= (1 - noise / lVar)
            plt.imshow(m, extent=e)
        """
        if self.dcube is None or self.dcube.shape[0] == 1:
            return None, None
        sh = self.shifts(filtered=filtered, verbose=verbose)
        amax = np.abs(np.asarray(sh)).max()
        if amax == 0:   # all shifts are zero, prevent SyntaxError in np.histogram2d()
            amax = 1
        # bin edges for square histogram centered at 0,0
        bins = np.arange(-amax - 0.5, amax + 1.5)
        extent = [-amax, amax, -amax, amax]
        h, _, _ = np.histogram2d(np.asarray(sh)[:, 0],
                                 np.asarray(sh)[:, 1],
                                 bins=bins)
        if verbose:
            peak_val = int(h.max())
            mx, my = np.where(h==np.amax(h))
            mx = int(bins[int(mx)] + 0.5)
            my = int(bins[int(my)] + 0.5)
            print('Shifts (filtered):') if filtered else print('Shifts (unfiltered):')
            print(f'   Range: {int(np.asarray(sh).min())} - {int(np.asarray(sh).max())}')
            print(f'   Maximum {peak_val} at ({max}, {my})')
        return h, extent

    def shifts(self, frames=None, filtered=False, verbose=False):
        """Calcultes frame shift by cross correlation of images (total intensity).

            Parameters
            ----------
               frames:     Iterable
                           Frame numbers for which shifts are calculated. First
                           frame given is used a reference.
                           Note, that the frame number denotes the index within
                           the data cube loaded. This is different from the
                           real frame number (stored in the `frame_list`
                           attribute) if only a subset of frames was loaded.

             filtered:     Bool
                           If True, use Wiener filtered data.
              verbose:     Bool
                           Provide additional info if set to True.

            Returns
            -------
                            List of tuples (dx, dy) containing the shift for
                            all frames or empty list if only a single frame
                            is present.
                            CAREFUL! Non-empty list ALWAYS contains 'meta.Sweeps'
                            elements and contains (0, 0) for frames that were
                            not in the list provided by keyword 'frames='.

            Examples
            --------
                # Get list of (possible) shifts [(dx0, dy0), (dx1, dx2), ...]
                # in pixels of individual frames using frame 0 as reference.
                # The shifts are calculated from the cross correlation of the
                # images of the total x-ray intensity of each individual frame.
                # Verbose output.
                >>>> dc.shifts(verbose=True)
                Frame 0 used a reference
                Average of (-2, -1) (0, 0) set to (-1, 0) in frame 24
                [(0, 0),
                 (0, 0),
                 (1, 1),
                 .
                 .
                 .
                 (0, -1)]

                # Use Wiener filtered images to calculate shifts.
                >>>> dc.shifts(filtered=True)
                /.../miniconda3/lib/python3.7/site-packages/scipy/signal/signaltools.py:1475: RuntimeWarning: divide by zero encountered in true_divide
                  res *= (1 - noise / lVar)
                /.../miniconda3/lib/python3.7/site-packages/scipy/signal/signaltools.py:1475: RuntimeWarning: invalid value encountered in multiply
                  res *= (1 - noise / lVar)
                [(0, 0),
                 (0, 0),
                 (1, 0),
                 .
                 .
                 .
                 (0, -1)]

                # Calculate shifts for selected frames (odd frames) only. In
                # this case `dc.drift_images[1]` (first frame given) is used
                # as reference.
                # Verbose output.
                >>>> dc.shifts(frames=range(1, dc.dcube.shape[0], 2), verbose=True)
                Frame 1 used a reference
                [(0, 0),
                 (0, 0),
                 (0, 0),
                 (0, 1),
                 (0, 0),
                 .
                 .
                 .
                 (0, 0),
                 (-1, -1)]
        """
        if self.dcube is None or self.dcube.shape[0] == 1:
            # only a single frame present
            return []
        if frames is None:
            frames = range(self.dcube.shape[0])
        # Always use first frame given as reference
        ref = wiener(self.map(frames=[frames[0]])) if filtered else self.map(frames=[frames[0]])
        shifts = [(0, 0)] * self.dcube.shape[0]
        if verbose:
            print(f'Frame {frames[0]} used a reference')
        for f in frames[1:]:    # skip reference frame
            c = correlate(ref, wiener(self.map(frames=[f]))) if filtered else correlate(ref, self.map(frames=[f]))
            # image size s=self.dcube.shape[1]
            # c has shape (2 * s - 1, 2 * s - 1)
            # Autocorrelation peaks at [s - 1, s - 1]
            # i.e. offset is at dy (dy) index_of_maximum - s + 1.
            dx, dy = np.where(c==np.amax(c))
            if dx.shape[0] > 1 and verbose:
                # Report cases where averaging was applied
                print('Average of', end=' ')
                for x, y in zip(dx, dy):
                    print(f'({x - self.dcube.shape[1] + 1}, '
                          f'{y - self.dcube.shape[1] + 1})',
                          end=' ')
                print(f'set to ({round(dx.mean() - self.dcube.shape[1] + 1)}, '
                      f'{round(dy.mean() - self.dcube.shape[1] + 1)}) '
                      f'in frame {f}')
            # More than one maximum is possible, use average
            dx = round(dx.mean())
            dy = round(dy.mean())
            shifts[f] = (dx - self.dcube.shape[1] + 1,
                         dy - self.dcube.shape[1] + 1)
        return shifts

    def map(self, interval=None, energy=False, frames=None, align='no',
            verbose=False):
        """Returns map corresponding to an interval in spectrum.

            Parameters
            ----------
                interval:   Tuple (number, number)
                            Defines interval (channels, or energy [keV]) to be
                            used for map. None implies that all channels are
                            integrated.
                  energy:   Bool
                            If false (default) interval is specified as channel
                            numbers otherwise (True) interval is specified as
                            'keV'.
                  frames:   Iterable (tuple, list, array, range object)
                            Frame numbers included in map. If split_frames is
                            active and frames is not specified all frames are
                            included.
                            Note, that the frame number denotes the index within
                            the data cube loaded. This is different from the
                            real frame number (stored in the `frame_list`
                            attribute) if only a subset of frames was loaded.

                   align:   Str
                            'no': Do not align individual frames.
                            'yes': Align frames (use unfiltered frames in
                                   cross correlation).
                            'filter': Align frames (use  Wiener filtered
                                      frames in cross correlation).
                 verbose:   Bool
                            If True, output some additional info.

            Returns
            -------
                map:   Ndarray
                       Spectral Map.

            Examples
            --------
                # Plot x-ray intensity integrated over all frames.
                >>>> plt.imshow(dc.map())
                <matplotlib.image.AxesImage at 0x7f7192ee6dd0>

                # Only use given interval of energy channels to calculate
                # map.
                >>>> plt.imshow(dc.map(interval=(115, 130)))
                <matplotlib.image.AxesImage at 0x7f7191eefd10>

                # Specify interval by energy [keV] instead of channel numbers.
                >>>>plt.imshow(dc.map(interval=(8,10), energy=True))
                <matplotlib.image.AxesImage at 0x7f4fd0616950>

                # If option 'split_frames' was used to read the data you can
                # plot the map of a single frame.
                >>>> plt.imshow(dc.map(frames=[3]))
                <matplotlib.image.AxesImage at 0x7f06c05ef750>

                # Map corresponding to the sum of a few selected frames.
                >>>> m = dc.map(frames=[3,5,11,12,13])

                # Cu Kalpha map of all even frames.
                >>>> m = dc.map(interval=(7.9, 8.1),
                                energy=True,
                                frames=range(0, dc.dcube.shape[0], 2))

                # Correct for frame shifts (calculated from unfiltered frames)
                # with verbose output.
                >>>> dc.map(align='yes', verbose=True)
                Using channels 0 - 4000
                Frame 0 used a reference
                Average of (-2, -1) (0, 0) set to (-1, 0) in frame 24

                # Cu Kalpha map of frames 0..10. Frames are aligned using
                # frame 5 as reference. Wiener filtered frames are used to
                # calculate the shifts.
                # Verbose output
                >>>>m = dc.map(interval=(7.9, 8.1),
                               energy=True,
                               frames=[5,0,1,2,3,4,6,7,8,9,10],
                               align='filter',
                               verbose=True)
                Using channels 790 - 810
                Frame 5 used a reference
                /home/alxneit/share/miniconda3/lib/python3.7/site-packages/scipy/signal/signaltools.py:1475: RuntimeWarning: divide by zero encountered in true_divide
                  res *= (1 - noise / lVar)
                /home/alxneit/share/miniconda3/lib/python3.7/site-packages/scipy/signal/signaltools.py:1475: RuntimeWarning: invalid value encountered in multiply
                  res *= (1 - noise / lVar)
        """
        # Check for valid keyword arguments
        assert align.lower() in ['yes', 'no', 'filter']

        if self.dcube is None:  # Only metadata was read
            return None

        if not interval:
            interval = (0, self.dcube.shape[3])
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

        if verbose:
            print(f'Using channels {interval[0]} - {interval[1]}')

        if interval[0] > self.dcube.shape[3] or interval[1] > self.dcube.shape[3]:
            warn(f'Interval {interval[0]}-{interval[1]} lies (partly) outside of data range 0-{self.dcube.shape[3]}')

        if self.dcube.shape[0] == 1:   # only a single frame (0) present
            return self.dcube[0, :, :, interval[0]:interval[1]].sum(axis=-1)

        # split_frame is active but no alignment required
        N = self.dcube.shape[1]     # image size
        if align == 'no':
            if frames is None:
                return self.dcube[:, :, :, interval[0]:interval[1]].sum(axis=(0, -1))
            # Only sum frames specified
            m = np.zeros((N, N))
            for frame in frames:
                m += self.dcube[frame, :, :, interval[0]:interval[1]].sum(axis=-1)
            return m

        # Alignment is required
        if frames is None:
            # Sum all frames
            frames = np.arange(self.dcube.shape[0])
        # Calculate frame shifts
        if align == 'filter':
            shifts = self.shifts(frames=frames, filtered=True, verbose=verbose)
        if align == 'yes':
            shifts = self.shifts(frames=frames, verbose=verbose)
        # Allocate array for result
        res = np.zeros((2*N, 2*N))
        x0 = N // 2
        y0 = N // 2
        for f in frames:
            # map of this frame summed over all energy intervals
            dx, dy = shifts[f]
            res[x0-dx:x0-dx+N, y0-dy:y0-dy+N] += \
                    self.dcube[f, :, :, interval[0]:interval[1]].sum(axis=-1)

        return res[x0:x0+N, y0:y0+N]

    def __correct_spectrum(self, s):
        """Apply non-linear energy correction at low energies to spectrum.

            Parameters
            ----------
                    s:  Ndarray
                        Uncorrected spectrum.

            Returns
            -------
                        Ndarray
                        Original or corrected spectrum, depending on whether
                        correction is necessary.
        """
        def apply_correction(s, ExCoef):
            """Applies the correction formula.

                Parameters
                ----------
                        s:  Ndarray
                            Original spectrum
                   ExCoef:  List
                            Correction coefficients.

                Returns
                -------
                        s:  Ndarray
                            Corrected spectrum.
            """
            CH_Res = self.parameters['PTTD Param'] \
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
            #     >>>> ref_spec[0:100].sum()                          #
            #     200468                                              #
            #     >>>> corrected_spec[0:100].sum()                    #
            #     200290                                              #
            #                                                         #
            ###########################################################
            E_corr = ExCoef[0]*E_uncorr**2 + ExCoef[1]*E_uncorr + ExCoef[2]
            s[0:N] = np.interp(E_uncorr, E_corr, s[0:N])
            return s

        Tpl_cond = self.parameters['EDS Data'] \
                                  ['AnalyzableMap MeasData']['Meas Cond'] \
                                  ['Tpl']
        try:
            ExCoef = self.parameters['PTTD Param'] \
                                    ['Params']['PARAMPAGE1_EDXRF']['Tpl'][Tpl_cond] \
                                    ['ExCoef']
            return apply_correction(s, ExCoef)
        except KeyError:
            return s

    def __spectrum_cROI(self, ROI, frames):
        """Returns spectrum integrated over a circular ROI

            Parameters
            ----------
                    ROI:    Tuple (center_x, center_y, radius)
                 frames:    Iterable (tuple, list, array, range object)
                            Frame numbers included in spectrum. If split_frames
                            is active and frames is not specified all frames
                            are included.

            Returns
            -------
               spectrum:    Ndarray
                            EDX spectrum.
        """
        # check validity of ROI
        min_x = ROI[0] - ROI[2]
        max_x = ROI[0] + ROI[2]
        min_y = ROI[1] - ROI[2]
        max_y = ROI[1] + ROI[2]
        if not all(0 <= val < self.dcube.shape[1] for val in [min_x, max_x, min_y, max_y]):
            raise ValueError(f"ROI {ROI} lies partially outside data cube")

        # Masking array with ones within circular ROI and zeros outside.
        # Only use slice containing the circle.
        N = 2 * ROI[2] + 1
        mask = np.zeros((N, N))
        x, y = np.ogrid[:N, :N]
        # Compare squares to avoid sqrt()
        r2 = (x + min_x - ROI[0])**2 + (y + min_y - ROI[1])**2
        m = r2 <= ROI[2]**2
        mask[m] = 1

        # Assemble list of frames
        if self.dcube.shape[0] == 1:    # Only a single frame present
            frames = [0]
        # Many frames are present
        if frames is None:  # No frames are specified explicitly, use all
            frames = range(self.dcube.shape[0])

        spectrum = np.zeros(self.dcube.shape[3])
        # iterate through all frames
        for frame in frames:
            # We have to mask the image at each energy
            for i in range(self.dcube.shape[3]):
                spectrum[i] += (mask * self.dcube[frame, min_x:max_x + 1, min_y:max_y + 1, i]).sum()

        return spectrum

    def spectrum(self, ROI=None, frames=None):
        """Returns spectrum integrated over a ROI.

            Parameters
            ----------
                     ROI:   Tuple (int, int)
                            Tuple (int, int, int)
                            Tuple (int, int, int, int)
                            or None.
                            Defines ROI for which spectrum is extracted.
                            None implies that the whole data cube is used.
                            A tuple (v, h) defines a single point ROI given by
                            its vertical and horizontal pixel index.
                            A tuple (center_v, center_h, radius) defines a
                            circular ROI including its boundary.
                            A tuple (top, bottom, left, right) defines a
                            rectangular ROI with boundaries included.
                            Numbers are pixel indices in the range 0 <= N < ImageSize.
                            Note, that this definition implies y-axis before
                            x-axis and the order of the numbers is the same as
                            when applied in a python slice ([top:bottom, left:right]).
                  frames:   Iterable (tuple, list, array, range object)
                            Frame numbers included in spectrum. If split_frames
                            is active and frames is not specified all frames
                            are included.
                            Note, that the frame number denotes the index within
                            the data cube loaded. This is different from the
                            real frame number (stored in the `frame_list`
                            attribute) if only a subset of frames was loaded.

            Returns
            -------
                spectrum:   Ndarray
                            EDX spectrum

            Examples
            --------
                # Plot spectrum integrated over full image.
                # If option 'split_frames' was used to read the data the
                # following plots spectra of all frames added together.
                >>>> plt.plot(dc.spectrum())
                [<matplotlib.lines.Line2D at 0x7f7192feec10>]

                # The integrated spectrum is also stored in the raw data and
                # can be accessed much quicker.
                >>>> plt.plot(dc.ref_spectrum)
                [<matplotlib.lines.Line2D at 0x7f3131a489d0>]

                # Plot spectrum corresponding to a single pixel. ROI is specified
                # as tuple (v, h) of pixel coordinatess.
                >>>> plt.plot(dc.spectrum(ROI=(45, 13)))
                <matplotlib.lines.Line2D at 0x7fd1423758d0>

                # Plot spectrum corresponding to a circular ROI specified
                # as tuple (center_v, center_h, radius) of pixel coordinatess.
                >>>> plt.plot(dc.spectrum(ROI=(80, 60, 15)))
                <matplotlib.lines.Line2D at 0x7fd14208f4d0>

                # Plot spectrum corresponding to a (rectangular) ROI specified
                # as tuple (top, bottom, left, right) of pixels.
                >>>> plt.plot(dc.spectrum(ROI=(10, 20, 50, 100)))
                <matplotlib.lines.Line2D at 0x7f7192b58050>

                # Plot spectrum for a single frame ('split_frames' used).
                >>>> plt.plot(dc.spectrum(frames=[23]))
                <matplotlib.lines.Line2D at 0x7f06b3db32d0>

                # Extract spectrum corresponding to a few frames added.
                >>>> spec = dc.spectrum(frames=[0,2,5,6])

                # Spectrum of all odd frames added.
                >>>> spec = dc.spectrum(frames=range(1, dc.dcube.shape[0], 2))
        """
        if self.dcube is None:  # Only metadata was read
            return None

        if not ROI:
            ROI = (0, self.dcube.shape[1] - 1, 0, self.dcube.shape[1] - 1)
        # ROI elements need to be ints
        if not all(isinstance(el, int) for el in ROI):
            raise ValueError(f"ROI {ROI} contains non-integer elements")
        if len(ROI) == 2:   # point ROI
            ROI = (ROI[0], ROI[0], ROI[1], ROI[1])
        if len(ROI) == 3:   # circular ROI, special
            return self.__correct_spectrum(self.__spectrum_cROI(ROI, frames))

        # check that ROI lies fully within the data cube
        if not all(0 <= val < self.dcube.shape[1] for val in ROI):
            raise ValueError(f"ROI {ROI} lies partially outside data cube")

        if self.dcube.shape[0] == 1:   # only a single frame (0) present
            s = self.dcube[0, ROI[0]:ROI[1] + 1, ROI[2]:ROI[3] + 1, :].sum(axis=(0, 1))
            return self.__correct_spectrum(s)

        # split_frames is active
        if frames is None:  # no frames specified, sum all frames
            s = self.dcube[:, ROI[0]:ROI[1] + 1, ROI[2]:ROI[3] + 1, :].sum(axis=(0, 1, 2))
            return self.__correct_spectrum(s)

        # only sum specified frames
        s = np.zeros(self.dcube.shape[3], dtype=self.dcube.dtype)
        for frame in frames:
            s += self.dcube[frame, ROI[0]:ROI[1] + 1, ROI[2]:ROI[3] + 1, :].sum(axis=(0, 1))
        return self.__correct_spectrum(s)

    def time_series(self, interval=None, energy=False, frames=None):
        """Returns x-ray intensity integrated in `interval` for all frames.

            Parameters
            ----------
                interval:   Tuple (number, number).
                            Defines interval (channels, or energy [keV]) to be
                            used for map. None implies that all channels are
                            integrated.
                  energy:   Bool.
                            If false (default) interval is specified as channel
                            numbers otherwise (True) interval is specified as
                            'keV'.
                  frames:   Iterable (tuple, list, array, range object).
                            Frame numbers included in time series (or None if
                            all frames are used). The integrated number of
                            counts is set to 'NaN' for all other frames.
                            Note, that the frame number denotes the index within
                            the data cube loaded. This is different from the
                            real frame number (stored in the `frame_list`
                            attribute) if only a subset of frames was loaded.


            Returns
            -------
                            Ndarray
                            Time evolution of intergrated intensity in interval.

            Examples
            --------
                # Intgrate carbon Ka peak (interval specified as channels)
                >>>> dc.time_series(interval=(20,40))
                array([1696., 1781., 1721., 1795., 1744., 1721., 1777., 1711., 1692.,
                       1752., 1651., 1664., 1693., 1696., 1736., 1682., 1707., 1710.,
                       1685., 1785., 1731., 1752., 1729., 1757., 1678., 1752., 1721.,
                       1740., 1696., 1718., 1737., 1740., 1719., 1670., 1692., 1649.,
                       1718., 1660., 1700., 1702., 1693., 1722., 1675., 1716., 1664.,
                       1761., 1691., 1731., 1663., 1669.])

                # Integrate oxygen Ka peak (interval specified as energy [keV])
                # and remove a few bad frames (11,12) from time series.
                >>>> frames = [f for f in range(dc.dcube.shape[0]) if f not in [11, 12]]
                >>>> dc.time_series(interval=(0.45, 0.6), energy=True, frames=frames)
                array([1042., 1128., 1032., 1016., 1031., 1019., 1070., 1014., 1078.,
                       1078., 1086.,   nan,   nan, 1025., 1028., 1040., 1084., 1020.,
                       1015., 1099., 1074., 1108., 1059., 1032., 1131., 1029., 1073.,
                       990., 1088., 1092., 1093., 1038., 1119., 1023., 1129., 1054.,
                       1072., 1051., 1039., 1048., 1062., 1099., 1063., 1092., 1073.,
                       1050., 1088., 1018., 1070., 1089.])

        """
        if self.dcube is None:  # Only metadata was read
            return None

        if not interval:
            interval = (0, self.dcube.shape[3])
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

        if interval[0] > self.dcube.shape[3] or interval[1] > self.dcube.shape[3]:
            warn(f'Interval {interval[0]}-{interval[1]} lies (partly) outside of data range 0-{self.dcube.shape[3]}')

        if frames is None:
            # For consistency, explicitly set dtype to 'float'. We need to
            # allow for NaN in unspecified frames in the else-clause below.
            ts = self.dcube[:, :, :, interval[0]:interval[1]].sum(axis=(1, 2, 3)).astype('float')
        else:
            ts = np.full((self.dcube.shape[0],), np.nan)
            for f in frames:
                ts[f] =self.dcube[f, :, :, interval[0]:interval[1]].sum(axis=(0, 1, 2))
        return ts

    def make_movie(self, fname=None, **kws):
        """Makes a movie of EDS data and drift_images

            Parameters
            ----------
                fname :     Str (or None)
                            Filename for movie file. If none is supplied the base
                            name of the '.pts' file is used.

            Returns
            -------
                            None.

            Examples
            --------
            >>>> dc = JEOL_pts('data/128.pts', split_frames=True, read_drift=True)

            # Make movie and store is as 'data/128.mp4'.
            >>>> dc.make_movie()
            # Only use Cu K_alpha line.
            >>>> dc.make_movie(interval=(7.9, 8.1), energy=True)

            # Make movie (one frame only, drift_image will be blank) and
            # save is as 'dummy.mp4'.
            >>>> dc = JEOL_pts('data/128.pts')
            >>>> dc.make_movie(fname='dummy.mp4')
        """
        if self.dcube is None:  # Only metadata was read
            return

        if fname is None:
            fname = os.path.splitext(self.file_name)[0] + '.mp4'

        # remove `frames=` keyword from dict as it would interfere later
        try:
            kws.pop('frames')
        except KeyError:
            pass

        # We might have read only a sublist of frames thus determine frames
        # loaded.
        frame_list = self.frame_list if self.frame_list else range(self.dcube.shape[0])

        # Maxima of the two type of images used to normalize images of both
        # series.
        try:
            STEM_max = max([self.drift_images[i].max() for i in frame_list])
        except TypeError:   # no drift_image available
            STEM_max = 1.0
        EDS_max = max([self.map(frames=[i]).max() for i in range(self.dcube.shape[0])])

        # Default dtype for maps is 'float64'. To minimize memory use select
        # smallest dtype possible.
        if EDS_max < 2**8:
            EDS_dtype = 'uint8'
        elif EDS_max < 2**16:
            EDS_dtype = 'uint16'
        elif EDS_max < 2**32:
            EDS_dtype = 'uint32'
        else:
            EDS_dtype = 'float64'

        # `self.drift_images.dtype` is 'uint16'. Select 'uint8' if possible.
        STEM_dtype = 'uint8'if STEM_max < 2**8 else 'uint16'

        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Note, more STEM images are present if only a subset of frames was
        # read.
        frames = []
        for i, STEM_i in enumerate(frame_list):
            EDS_map = self.map(frames=[i], **kws).astype(EDS_dtype)
            try:
                STEM_image = self.drift_images[STEM_i].astype(STEM_dtype)
            except TypeError:   # no drift_image available, dummy image
                STEM_image = np.full_like(EDS_map, np.nan).astype('uint8')
            image = np.concatenate((STEM_image / STEM_max, EDS_map / EDS_max),
                                   axis=1)
            frame = plt.imshow(image, animated=True)
            # Add frame number. Use index of STEM image in case only a subset
            # of frames was read.
            text = ax.annotate(STEM_i, (1, -5), annotation_clip=False)
            frames.append([frame, text])

        ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                        repeat_delay=1000)
        ani.save(fname)

    def save_dcube(self, fname=None):
        """Saves (compressed) data cube.

            Parameters
            ----------
                fname:  Str (or None)
                        Filename. If none is supplied the base name of the
                        '.pts' file is used.

            Examples
            --------
                # Save extracted data cube. File name is the same as the '.pts'
                # file but extension is changed to 'npz'.
                # This makes only sense if option 'split_frames' was NOT used
                # to read the data. Otherwise the file size is larger than the
                #'.pts' file and saving becomes time consuming.
                >>>> dc.save_dcube()

                # You can also supply your own filename, but use '.npz' as
                # extension.
                >>>> dc.save_dcube(fname='my_new_filename.npz')

                # If you want to read the data cube into your own program.
                >>>> npzfile = np.load('128.npz')
                >>>> dcube = npzfile['arr_0']

                # split_frames was not active when data was saved.
                >>>> dcube.shape
                (1, 128, 128, 4000)

                # Split_frames was active when data was saved.
                >>>> dcube.shape
                (50, 128, 128, 4000)
        """
        if self.dcube is None:  # Only metadata was read
            return

        if fname is None:
            fname = os.path.splitext(self.file_name)[0] + '.npz'
        np.savez_compressed(fname, self.dcube)

    def __load_dcube(self, fname):
        """Loads a previously saved data cube.

            Parameters
            ----------
                fname:  Str
                        File name of '.npz' file (must end in '.npz').
        """
        self.file_name = fname
        self.file_date = None
        self.frame_list = None
        self.drift_images = None
        npzfile = np.load(fname)
        self.dcube = npzfile['arr_0']

    def save_hdf5(self, fname=None, **kws):
        """Saves all data including attributes to hdf5 file

            Parameters
            ----------
                fname:  Str
                        File name of '.h5' file (must end in '.h5').
                        If none is supplied the base name of the
                        '.pts' file is used.

            Examples
            --------
                # Save data with file name based on the '.pts' file but
                # extension is changed to 'h5'.
                >>>> dc.save_hdf5()

                # You can also supply your own filename, but use '.h5' as
                # extension.
                >>>> dc.save_hdf5(fname='my_new_filename.h5')

                # Pass along keyword arguments such as e.g. compression
                >>>> dc.save_hdf5('compressed.h5',
                                  compression='gzip',
                                  compression_opts=9)

                # If you want to read the data cube into your own program.
                >>>> hf = h5py.File('my_file.h5', 'r')

                # List data sets available
                >>>> for name in hf:
                         print(name)
                EDXRF
                dcube
                drift_images

                # Use '[()]' to get actual data not just a reference to
                # stored data in file
                >>>> hf['EDXRF'][()]
               array([    0,     0,     0,     0,   876,   719,   339,   531,   904,
                       1223,  1268,  1099,   899,   695,   621,   584,   519,   525,
                        .
                        .
                         22,    33,    23,    20,    20,    15,    29,    27,    17,
                         19], dtype=int32)

                # List attributes
                >>>> print(hf1.attrs.keys())
                <KeysViewHDF5 ['file_date', 'file_name', 'parameters']>
                >>>> hf.attrs['file_name']
                'data/128.pts'
        """
        if self.dcube is None:  # Only metadata was read
            return

        if fname is None:
            fname = os.path.splitext(self.file_name)[0] + '.h5'

        with h5py.File(fname, 'w') as hf:
            hf.create_dataset('dcube', data=self.dcube, **kws)
            if self.drift_images is not None:
                hf.create_dataset('drift_images', data=self.drift_images, **kws)

            hf.attrs['file_name'] = self.file_name
            hf.attrs['file_date'] = self.file_date
            if self.frame_list is not None:
                hf.attrs['frame_list'] = self.frame_list
            # avoid printing of ellipsis in arrays / lists
            np.set_printoptions(threshold=sys.maxsize)
            hf.attrs['parameters'] = str(self.parameters)

    def __load_hdf5(self, fname):
        """Loads data including attributes from hdf5 file

            Parameters
            ----------
                fname:  Str
                        File name of '.h5' file (must end in '.h5').
        """
        with h5py.File(fname, 'r') as hf:
            self.dcube = hf['dcube'][()]

            self.drift_images = hf['drift_images'][()] if 'drift_images' in hf.keys() else None
            self.frame_list = hf['drift_images'][()] if 'frame_list' in hf.keys() else None

            self.file_date = hf.attrs['file_date']
            self.file_name = hf.attrs['file_name']

            aeval = asteval.Interpreter()
            self.parameters = aeval(hf.attrs['parameters'])



class JEOL_image():
    """Read JEOL image data ('.img' and '.map' files).

        Parameters
        ----------
            fname:      Str
                        Filename.

        Examples
        --------
        >>>> from JEOL_eds import JEOL_image

        >>>> demo = JEOL_image('data/demo.img')
        >>>> demo.file_name
        'data/demo.img'

        >>>> demo.file_date
        '2021-08-13 16:09:06'

        # Meta data stored in file.
        >>>> demo.parameters
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
        >>>> plt.imshow(demo.image)
        <matplotlib.image.AxesImage at 0x7fa08425d350>

        # Read a map file.
        >>>> demo = JEOL_image('data/demo.map')

        # Print calibration data (pixel size in nm).
        # This is only available for '*.map' files.
        >>>> demo.pixel_size
        0.99
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
            self.fileformat = decode(fd.read(32).rstrip(b"\x00"))
            head_pos, head_len, data_pos = np.fromfile(fd, "<I", 3)
            fd.seek(data_pos + 12)
            self.parameters = parsejeol(fd)
            self.file_date = str(datetime(1899, 12, 30) + timedelta(days=self.parameters["Image"]["Created"]))

            # Make image data easier accessible
            sh = self.parameters["Image"]["Size"]
            self.image = self.parameters["Image"]["Bits"]
            self.image.resize(tuple(sh))

            # Nominal pixel size in nm
            #
            # In '.img' files this is only correct if its dimensions coincide with the '.map' files
            # that are recorded simultaneously to the edx data. These are typically the first two
            # '.map' files in the list of the project.
            #
            if os.path.splitext(fname)[1] == '.map':
                self.pixel_size = self.parameters["Instrument"]["ScanSize"] / self.parameters["Instrument"]["Mag"] * 1000.0
