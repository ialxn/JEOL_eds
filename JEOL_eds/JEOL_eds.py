#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from scipy.signal import wiener, correlate


class JEOL_pts:
    """Work with JEOL '.pts' files

        Examples
        --------

        >>>> from JEOL_eds import JEOL_pts

        # Initialize JEOL_pts object (read data from '.pts' file).
        # Data cube has dtype = 'uint16' (default).
        >>>> dc = JEOL_pts('128.pts')
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

        # Useful attributes.
        >>>> dc.file_name
        '128.pts'               # File name loaded from.
        >>>> dc.file_date
        '2020-10-23 11:18:40'   # File creation date
        >>>> dc.dcube.shape     # Shape of data cube
        (1, 128, 128, 4000)

        # Store individual frames.
        >>>> dc=JEOL_pts('128.pts', split_frames=True)
        >>>> dc.dcube.shape
        (50, 128, 128, 4000)

        # Also read and store BF images (one per frame) present if
        # option "correct for sample movement" was active during
        # data collection.
        # Attribute will be set to 'None' if no data was found!
        >>> dc = JEOL_pts('128.pts', read_drift=True)
        dc.drift_images is None
        False
        >>> dc.drift_images.shape
        (50, 128, 128)

        >>>> import matplotlib.pyplot as plt
        plt.imshow(dc.drift_images[0])
        <matplotlib.image.AxesImage at 0x7ff3e9976550>

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

        # Use all energy channels, i.e. plot map of total number of counts.
        # If option 'split_frames' was used to read the data, the following
        # draws maps for all frames added together.
        >>>> plt.imshow(dc.map())
        <matplotlib.image.AxesImage at 0x7f7192ee6dd0>
        # Specify energy interval (channels containing a spectral line) to
        # be used for map. Used to map specific elements.
        >>>> plt.imshow(dc.map(interval=(115, 130)))
        <matplotlib.image.AxesImage at 0x7f7191eefd10>
        # Specify interval by energy (keV) instead of channel numbers.
        >>>>plt.imshow(dc.map(interval=(8,10), energy=True))
        <matplotlib.image.AxesImage at 0x7f4fd0616950>
        # If option 'split_frames' was used to read the data you can plot
        # the map of a single frame.
        >>>> plt.imshow(dc.map(frames=[3]))
        <matplotlib.image.AxesImage at 0x7f06c05ef750>
        # Map correponding to a few selected frames.
        >>>> m = dc.map(frames=[3,5,11,12,13])
        # Cu Kalpha map of all even frames
        >>>> m = dc.map(interval=(7.9, 8.1),
                        energy=True,
                        frames=range(0, dc.dcube.shape[0], 2))
        # Correct for frame shifts (with additional output).
        >>>> dc.map(align='yes', verbose=True)
        Using channels 0 - 4000
        Frame 0 used a reference
        Average of (-2, -1) (0, 0) set to (-1, 0) in frame 24

        # Plot spectrum integrated over full image. If option 'split_frames'
        # was used to read the data the following plots spectra for all frames
        added together.
        >>>> plt.plot(dc.spectrum())
        [<matplotlib.lines.Line2D at 0x7f7192feec10>]
        # The integrated spectrum is also stored in the raw data and can be
        # accessed much quicker.
        >>>> plt.plot(dc.ref_spectrum)
        [<matplotlib.lines.Line2D at 0x7f3131a489d0>]

        # Plot spectrum corresponding to a (rectangular) ROI specified as
        # tuple (left, right, top, bottom) of pixels.
        >>>> plt.plot(dc.spectrum(ROI=(10,20,50,100)))
        <matplotlib.lines.Line2D at 0x7f7192b58050>
        # Plot spectrum for a single frame ('split_frames' used).
        >>>> plt.plot(dc.spectrum(frames=[23]))
        <matplotlib.lines.Line2D at 0x7f06b3db32d0>
        # Extract spectrum corresponding to a few frames added.
        >>>> spec = dc.spectrum(frames=[0,2,5,6])
        # Spectrum of all odd frames added.
        >>>> spec = dc.spectrum(frames=range(1, dc.dcube.shape[0], 2))

        # Save extracted data cube. File name is the same as the '.pts' file
        # but extension is changed to 'npz'.
        # This makes only sense if option 'split_frames' was used to read
        # the data as otherwise the file size is larger than the '.pts' file
        # and saving becomes time consuming.
        >>>> dc.save_dcube()
        # You can also supply your own filename, but use '.npz' as extension.
        >>>> dc.save_dcube(fname='my_new_filename.npz')

        # JEOL_pts object can also be initialized from a saved data cube. In
        # this case, dtype of the data cube is the same as in the stored data
        # and a possible 'dtype=' keyword is ignored.
        # This only initializes the data cube. Most attributes are not loaded
        # and are set to 'None'
        >>>> dc2 = JEOL_pts('128.npz')
        >>>> dc2.file_name
        '128.npz'
        >>>> dc2.parameters is None
        True

        # Get list of (possible) shifts [(dx0, dy0), (dx1, dx2), ...] in pixels
        # of individual frames using frame 0 as reference. The shifts are
        # calculated from the cross correlation of images of total intensity of
        # each individual frame.
        # Set "filtered=True" to calculate shifts from Wiener filtered images.
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

        # Get the 2D frequency distribution of the frames shifts using (or not)
        # Wiener filtered frames.
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

        >>>> m, e = dc.drift_statistics(filtered=True)
        /.../scipy/signal/signaltools.py:1475: RuntimeWarning: divide by zero encountered in true_divide
        res *= (1 - noise / lVar)
        /.../scipy/signal/signaltools.py:1475: RuntimeWarning: invalid value encountered in multiply
        res *= (1 - noise / lVar)
        plt.imshow(m, extent=e)

        # Calulate shifts for selected frames (odd frames) only. In this case
        # the first frame given is used as reference.
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

    def __init__(self, fname, dtype='uint16',
                 split_frames=False, E_cutoff=False, read_drift=False,
                 verbose=False):
        """Reads datacube from JEOL '.pts' file or from previously saved data cube.

            Parameters
            ----------
                 fname:     Str
                            Filename.
                 dtype:     Str
                            Data type used to store (not read) datacube.
                            Can be any of the dtype supported by numpy.
                            If a '.npz' file is loaded, this parameter is
                            ignored and the dtype corresponds to the one
                            of the data cube when it was stored.
          split_frames:     Bool
                            Store individual frames in the data cube (if
                            True), otherwise add all frames and store in
                            a single frame (default).
              E_cutoff:     Float
                            Energy cutoff in spectra. Only data below E_cutoff
                            are read.
            read_drift:     Bool
                            Read BF images (one BF image per frame stored in
                            the raw data, if the option "correct for sample
                            movement" was active while the data was collected).
               verbose:     Bool
                            Turn on (various) output.
        """
        self.split_frames = split_frames
        if os.path.splitext(fname)[1] == '.npz':
            self.parameters = None
            self.__load_dcube(fname)
        else:
            self.file_name = fname
            self.parameters, data_offset = self.__parse_header(fname)
            self.dcube = self.__get_data_cube(dtype, data_offset,
                                              E_cutoff=E_cutoff, verbose=verbose)
        if self.parameters:
            self.ref_spectrum = self.parameters['EDS Data']['AnalyzableMap MeasData']['Data']['EDXRF'][0:self.dcube.shape[3]]
        else:
            self.ref_spectrum = None

        if read_drift and os.path.splitext(fname)[1] == '.pts':
            self.drift_images = self.__read_drift_images(fname)
        else:
            self.drift_images = None

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
                    Copied almost verbatiom from Hyperspy (hyperspy/io_plugins/jeol.py).
        """
        with open(fname, "br") as fd:
            file_magic = np.fromfile(fd, "<I", 1)[0]
            assert file_magic == 304
            _ = fd.read(8).rstrip(b"\x00").decode("utf-8")
            _, _, head_pos, head_len, data_pos, data_len = np.fromfile(fd, "<I", 6)
            fd.read(128).rstrip(b"\x00").decode("utf-8")
            _ = fd.read(132).rstrip(b"\x00").decode("utf-8")
            self.file_date = str(datetime(1899, 12, 30) + timedelta(days=np.fromfile(fd, "d", 1)[0]))
            fd.seek(head_pos + 12)
            return self.__parsejeol(fd), data_pos

    @staticmethod
    def __parsejeol(fd):
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
                    Copied almost verbatiom from Hyperspy (hyperspy/io_plugins/jeol.py).
        """
        jTYPE = {
            1: "B",
            2: "H",
            3: "i",
            4: "f",
            5: "d",
            6: "B",
            7: "H",
            8: "i",
            9: "f",
            10: "d",
            11: "?",
            12: "c",
            13: "c",
            14: "H",
            20: "c",
            65553: "?",
            65552: "?",
            }

        final_dict = {}
        tmp_list = []
        tmp_dict = final_dict
        mark = 1
        while abs(mark) == 1:
            mark = np.fromfile(fd, "b", 1)[0]
            if mark == 1:
                str_len = np.fromfile(fd, "<i", 1)[0]
                kwrd = fd.read(str_len).rstrip(b"\x00")
                if (
                    kwrd == b"\xce\xdf\xb0\xc4"
                ):  # correct variable name which might be 'Port'
                    kwrd = "Port"
                elif (
                    kwrd[-1] == 222
                ):  # remove undecodable byte at the end of first ScanSize variable
                    kwrd = kwrd[:-1].decode("utf-8")
                else:
                    kwrd = kwrd.decode("utf-8")
                val_type, val_len = np.fromfile(fd, "<i", 2)
                tmp_list.append(kwrd)
                if val_type == 0:
                    tmp_dict[kwrd] = {}
                else:
                    c_type = jTYPE[val_type]
                    arr_len = val_len // np.dtype(c_type).itemsize
                    if c_type == "c":
                        value = fd.read(val_len).rstrip(b"\x00")
                        value = value.decode("utf-8").split("\x00")
                        # value = os.path.normpath(value.replace('\\','/')).split('\x00')
                    else:
                        value = np.fromfile(fd, c_type, arr_len)
                    if len(value) == 1:
                        value = value[0]
                    if kwrd[-5:-1] == "PAGE":
                        kwrd = kwrd + "_" + value
                        tmp_dict[kwrd] = {}
                        tmp_list[-1] = kwrd
                    elif kwrd in ("CountRate", "DeadTime"):
                        tmp_dict[kwrd] = {}
                        tmp_dict[kwrd]["value"] = value
                    elif kwrd == "Limits":
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
                        tmp_dict[kwrd]["index"] = value
                    else:
                        tmp_dict[kwrd] = value
                if kwrd == "Limits":
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

    def __CH_offset_from_meta(self):
        """Returns offset (channel corresponding to E=0).
        """
        Tpl_cond = self.parameters['EDS Data']['AnalyzableMap MeasData']['Meas Cond']['Tpl']
        return self.parameters['PTTD Param']['Params']['PARAMPAGE1_EDXRF']['Tpl'][Tpl_cond]['DigZ']

    def __get_data_cube(self, dtype, offset,
                        E_cutoff=None, verbose=False):
        """Returns data cube (F x X x Y x E).

            Parameters
            ----------
                dtype:      Str
                            Data type used to store data cube in memory.
                hsize:      Int
                            Number of header bytes.
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
        NumCH = self.parameters['PTTD Param']['Params']['PARAMPAGE1_EDXRF']['NumCH']
        ScanLine = self.parameters['PTTD Data']['AnalyzableMap MeasData']['Doc']['ScanLine']
        if E_cutoff:
            CoefA = self.parameters['PTTD Data']['AnalyzableMap MeasData']['Doc']['CoefA']
            CoefB = self.parameters['PTTD Data']['AnalyzableMap MeasData']['Doc']['CoefB']
            N_spec = round((E_cutoff - CoefB) / CoefA)
        else:
            N_spec = NumCH - CH_offset
        with open(self.file_name, 'rb') as f:
            f.seek(offset)
            data = np.fromfile(f, dtype='u2')
        if self.split_frames:
            Sweep = self.parameters['PTTD Data']['AnalyzableMap MeasData']['Doc']['Sweep']
            dcube = np.zeros([Sweep, ScanLine, ScanLine, N_spec], dtype=dtype)
        else:
            dcube = np.zeros([1, ScanLine, ScanLine, N_spec], dtype=dtype)
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
        scale = 4096 / ScanLine
        # map the size x size image into 4096x4096
        for d in data:
            N += 1
            if 32768 <= d < 36864:
                y = int((d - 32768) / scale)
            elif 36864 <= d < 40960:
                d = int((d - 36864) / scale)
                if self.split_frames and d < x:
                    # A new frame starts once the slow axis (x) restarts. This
                    # does not necessary happen at x=zero, if we have very few
                    # counts and nothing registers on first scan line.
                    frame += 1
                x = d
            elif 45056 <= d < END:
                z = int(d - 45056)
                z -= CH_offset
                if N_spec > z >= 0:
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
            print('Unidentified data items ({} out of {}, {:.2f}%) found:'.format(N, N_err, 100*N_err/N))
            for key in sorted(unknown):
                print('\t{}: found {} times'.format(key, unknown[key]))
        return dcube

    def __read_drift_images(self, fname):
        """Read BF images stored (option "correct for sample movement" was active)

            Parameters
            ----------
                fname:      Str
                            Filename.

            Returns
            -------
                ndarray or None if data is not available
                Stack of images with shape (N_images, im_size, im_size)

        Notes
        -----
            Based on a code fragment by @sempicor at
            https://github.com/hyperspy/hyperspy/pull/2488
        """
        ScanLine = self.parameters["PTTD Data"]["AnalyzableMap MeasData"]["Doc"]["ScanLine"]
        with open(fname) as f:
            f.seek(8*16**3)     # data seems to be at fixed offset
            rawdata = np.fromfile(f, dtype='u2')
            ipos = np.where(np.logical_and(rawdata >= 40960, rawdata < 45056))[0]
            if len(ipos) == 0:  # No data available
                return None
            I = np.array(rawdata[ipos]-40960, dtype='uint16')
            N_images = int(np.ceil(ipos.shape[0] / ScanLine**2))
            return I.reshape((N_images, ScanLine, ScanLine))

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
        """
        if self.dcube.shape[0] == 1:
            return None, None
        sh = self.shifts(filtered=filtered, verbose=verbose)
        amax = np.abs(np.asarray(sh)).max()
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
            if filtered:
                print('Shifts (filtered):')
            else:
                print('Shifts (unfiltered):')
            print('   Range: {} - {}'.format(int(np.asarray(sh).min()),
                                             int(np.asarray(sh).max())))
            print('   Maximum {} at ({}, {})'.format(peak_val, mx, my))
        return h, extent

    def shifts(self, frames=None, filtered=False, verbose=False):
        """Calcultes frame shift by cross correlation of images (total intensity).

            Parameters
            ----------
               frames:     Iterable
                           Frame numbers for which shifts are calculated. First
                           frame given is used a reference.
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
        """
        if self.dcube.shape[0] == 1:
            # only a single frame present
            return []
        if frames is None:
            frames = range(self.dcube.shape[0])
        # Always use first frame given as reference
        if filtered:
            ref = wiener(self.map(frames=[frames[0]]))
        else:
            ref = self.map(frames=[frames[0]])
        shifts = [(0, 0)] * self.dcube.shape[0]
        if verbose:
            print('Frame {} used a reference'.format(frames[0]))
        for f in frames[1:]:    # skip reference frame
            if filtered:
                c = correlate(ref, wiener(self.map(frames=[f])))
            else:
                c = correlate(ref, self.map(frames=[f]))
            # image size s=self.dcube.shape[1]
            # c has shape (2 * s - 1, 2 * s - 1)
            # Autocorrelation peaks at [s - 1, s - 1]
            # i.e. offset is at dy (dy) index_of_maximum - s + 1.
            dx, dy = np.where(c==np.amax(c))
            if dx.shape[0] > 1 and verbose:
                # Report cases where averging was applied
                print('Average of', end=' ')
                for x, y in zip(dx, dy):
                    print('({}, {})'.format(x - self.dcube.shape[1] + 1,
                                            y - self.dcube.shape[1] + 1),
                          end=' ')
                print('set to ({}, {}) in frame {}'.format(round(dx.mean() - self.dcube.shape[1] + 1),
                                                           round(dy.mean() - self.dcube.shape[1] + 1),
                                                            f))
            # More than one maximum is possible, use average
            dx = round(dx.mean())
            dy = round(dy.mean())
            shifts[f] = (dx - self.dcube.shape[1] + 1, dy - self.dcube.shape[1] + 1)
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
                   align:   Str
                            'no': Do not aligne individual frames.
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
        """
        # Check for valid keyword arguments
        assert ['yes', 'no', 'filter'].index(align)

        if not interval:
            interval = (0, self.dcube.shape[3])
        if energy:
            CoefA = self.parameters['PTTD Data']['AnalyzableMap MeasData']['Doc']['CoefA']
            CoefB = self.parameters['PTTD Data']['AnalyzableMap MeasData']['Doc']['CoefB']
            interval = (int(round((interval[0] - CoefB) / CoefA)),
                        int(round((interval[1] - CoefB) / CoefA)))
        if verbose:
            print('Using channels {} - {}'.format(interval[0], interval[1]))

        if not self.split_frames:   # only a single frame (0) present
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
            res[x0-dx:x0-dx+N, y0-dy:y0-dy+N] += self.dcube[f, :, :,
                                                            interval[0]:interval[1]].sum(axis=-1)
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
            E_uncorr = np.arange(0, ExCoef[3], 0.01)
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
            # With 'test/128.pts' as example:                         #                             #
            #     >>>> ref_spec[0:100].sum()                          #                         #
            #     200468                                              #
            #     >>>> corrected_spec[0:100].sum()                    #                                #
            #     200290                                              #
            #                                                         #
            ###########################################################
            E_corr = ExCoef[0]*E_uncorr**2 + ExCoef[1]*E_uncorr + ExCoef[2]
            s[0:N] = np.interp(E_uncorr, E_corr, s[0:N])
            return s

        Tpl_cond = self.parameters['EDS Data']['AnalyzableMap MeasData']['Meas Cond']['Tpl']
        try:
            ExCoef = self.parameters['PTTD Param']['Params']['PARAMPAGE1_EDXRF']['Tpl'][Tpl_cond]['ExCoef']
            return apply_correction(s, ExCoef)
        except KeyError:
            return s

    def spectrum(self, ROI=None, frames=None):
        """Returns spectrum integrated over a ROI.

            Parameters
            ----------
                     ROI:   Tuple (int, int, int, int) or None
                            Defines ROI for which spectrum is extracted. ROI is
                            defined by its boundaries (left, right, top, bottom).
                            None implies that the whole image is used.
                  frames:   Iterable (tuple, list, array, range object)
                            Frame numbers included in spectrum. If split_frames
                            is active and frames is not specified all frames
                            are included.

            Returns
            -------
                spectrum:   Ndarray
                            EDX spectrum
        """
        if not ROI:
            ROI = (0, self.dcube.shape[1], 0, self.dcube.shape[1])
        if not self.split_frames:   # only a single frame (0) present
            s = self.dcube[0, ROI[0]:ROI[1], ROI[2]:ROI[3], :].sum(axis=(0, 1))
            return self.__correct_spectrum(s)

        # split_frames is active
        if frames is None:  # no frames specified, sum all frames
            s = self.dcube[:, ROI[0]:ROI[1], ROI[2]:ROI[3], :].sum(axis=(0, 1, 2))
            return self.__correct_spectrum(s)

        # only sum specified frames
        s = np.zeros(self.dcube.shape[-1])
        for frame in frames:
            s += self.dcube[frame, ROI[0]:ROI[1], ROI[2]:ROI[3], :].sum(axis=(0, 1))
        return self.__correct_spectrum(s)

    def save_dcube(self, fname=None):
        """Saves (compressed) data cube.

            Parameters
            ----------
                fname:  Str (or None)
                        Filename. If none is supplied the base name of the
                        '.pts' file is used.
        """
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
        npzfile = np.load(fname)
        self.dcube = npzfile['arr_0']
        if self.dcube.shape[0] > 1:
            self.split_frames = True
