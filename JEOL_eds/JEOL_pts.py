#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=C0103,C0301
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
import sys
from datetime import datetime, timedelta
from warnings import warn
import h5py
import asteval
import numpy as np
from scipy.signal import wiener, correlate
import matplotlib.pyplot as plt
from matplotlib import animation

from JEOL_eds.misc import _parsejeol, _correct_spectrum
from JEOL_eds.utils import rebin


class JEOL_pts:
    """Work with JEOL '.pts' files corresponding to elemental maps.

    Parameters
    ----------
    fname : Str
        Filename.
    dtype : Str
        Data type used to store (not read) data cube. Can be any of the dtypes
        supported by numpy. If a '.npz' file is loaded, this parameter is
        ignored and the dtype corresponds to the one of the data cube when it
        was stored.
    split_frames : Bool
        Store individual frames in the data cube (if True), otherwise add all
        frames and store in a single frame (default).
    frame_list : List (or None)
        List of frames to be read if split_frames was specified. Default (None)
        implies all frames present in data are read.
    E_cutoff : Float
        Energy cutoff in spectra. Only data below E_cutoff are read.
    read_drift : Bool
        Read BF images (one BF image per frame stored in the raw data, if the
        option "correct for sample movement" was active while the data was
        collected). All images are read even if only a subset of frames is read
        (frame_list is specified).
    only_metadata : Bool
        Only meta data is read (True) but nothing else. All other keywords are
        ignored.
    verbose : Bool
        Turn on (various) output.

    Notes
    -----
        JEOL's Analysis Station stores edx data collected for a single
        (horizontal) scan line in the same format. For these data use
        ``JEOL_eds.JEOL_DigiLine()``.

    Examples
    --------
    >>> from JEOL_eds import JEOL_pts
    >>> import JEOL_eds.utils as JU

    Initialize JEOL_pts object (read data from '.pts' file). Data cube has
    dtype = 'uint16' (default):
    >>> dc = JEOL_pts('data/128.pts')
    >>> dc.dcube.shape     # Shape of data cube
    (1, 128, 128, 4000)
    >>> dc.dcube.dtype
    dtype('uint16')

    Same, but specify dtype to be used for data cube:
    >>> dc = JEOL_pts('data/128.pts', dtype='int')
    >>> dc.dcube.dtype
    dtype('int64')

    Provide additional (debug) output when loading:
    >>> dc = JEOL_pts('data/128.pts', dtype='uint16', verbose=True) #doctest: +NORMALIZE_WHITESPACE
    Unidentified data items (82810 out of 2081741, 3.98%) found:
        24576: found 41858
        28672: found 40952

    Store individual frames:
    >>> dc = JEOL_pts('data/128.pts', split_frames=True)
    >>> dc.dcube.shape
    (50, 128, 128, 4000)

    For large data sets, read only a subset of frames:
    >>> small_dc = JEOL_pts('data/128.pts',
    ...                     split_frames=True, frame_list=[1,2,4,8,16])
    >>> small_dc.frame_list
    [1, 2, 4, 8, 16]

    >>> small_dc.dcube.shape
    (5, 128, 128, 4000)

    The frames in the data cube correspond to the original frames 1, 2, 4, 8,
    and 16.

    Only import spectrum up to cutoff energy [keV]:
    >>> dc = JEOL_pts('data/128.pts', E_cutoff=10.0)
    >>> dc.dcube.shape
    (1, 128, 128, 1000)

    Also read and store BF images (one per frame) present Attribute will be set
    to 'None' if no data was not read or found:
    >>> dc = JEOL_pts('data/128.pts', split_frames=True,
    ...               read_drift=True, frame_list=[1,2,4,8,16])
    >>> dc.drift_images is None
    False

    >>> dc.drift_images.shape
    (50, 128, 128)

    If only a subset of frames was read select the corresponding BF images
    as follows:
    >>> drift_images = [dc.drift_images[i] for i in dc.frame_list]

    Useful attributes:
    File name loaded from:
    >>> dc.file_name
    'data/128.pts'

    File creation date:
    >>> dc.file_date
    '2020-10-23 11:18:40'

     Mag calibration [nm / pixel]
    >>> dc.nm_per_pixel
    np.float64(1.93359375)

    More info is stored in attribute `JEOL_pts.parameters`:
    >>> p = dc.parameters

    Measurement parameters active when map was acquired:
    >>> meas = p['EDS Data']['AnalyzableMap MeasData']['Doc']
    >>> meas['LiveTime']
    np.float64(409.5)

    >>> meas['RealTime']
    np.float64(418.56)

    JEOL_pts objects can also be initialized from a saved data cube. In this
    case, the dtype of the data cube is the same as in the stored data and a
    possible 'dtype=' keyword is ignored. This only initializes the data cube.
    Most attributes are not loaded and are set to 'None'.

    First save data:
    >>> dc.save_dcube(fname='data/128.npz')

    Now load it again:
    >>> dc2 = JEOL_pts('data/128.npz')
    >>> dc2.file_name
    'data/128.npz'

    >>> dc2.parameters is None
    True

    Additionally, JEOL_pts object can be saved as hdf5 files. This has the
    benefit that all attributes (drift_images, parameters) are also stored.
    Use base name of original file and pass along keywords to
    `h5py.create_dataset()`:
    >>> dc.save_hdf5(fname='128.h5',
    ...              compression='gzip', compression_opts=9)

    Initialize from hdf5 file. Only filename is used, additional keywords are
    ignored.
    >>> dc3 = JEOL_pts('128.h5')

    Fast way to read and plot reference spectrum.
    >>> JU.plot_spectrum(JEOL_pts('data/64.pts', only_metadata=True).ref_spectrum)
    """

    def __init__(self, fname, dtype='uint16',
                 split_frames=False, frame_list=None,
                 E_cutoff=False, read_drift=False,
                 rebin=None, only_metadata=False, verbose=False):
        """Reads data cube from JEOL '.pts' file or from previously saved data cube.

        Parameters
        ----------
        fname : Str
            Filename.
        dtype : Str
            Data type used to store (not read) data cube. Can be any of the
            dtypes supported by numpy. If a '.npz' file is loaded, this
            parameter is ignored and the dtype corresponds to the one of the
            data cube when it was stored.
        split_frames : Bool
            Store individual frames in the data cube (if True), otherwise sum
            all frames and store in a single frame (default).
        frame_list : List
            List of frames to be read if split_frames was specified. Default
            (None) implies all frames present in data are read.
        E_cutoff : Float
            Energy cutoff in spectra. Only data below E_cutoff are read.
        read_drift : Bool
            Read BF images (one BF image per frame stored in the raw data, if
            the option "correct for sample movement" was active while the data
            was collected). All images are read even if only a subset of frames
            is read (frame_list is specified).
        rebin : Tuple
            Rebin drift images and data while reading the '.pts' file
            by (nw, nh). The integers nw and nh must be compatible with
            the scan size.
            This option is not used when reading '.npz' or '.h5' files.
        only_metadata : Bool
            Only metadata are read (True) but nothing else. All other keywords
            are ignored.
        verbose : Bool
            Turn on (various) output.
        """
        if os.path.splitext(fname)[1] == '.pts':
            self.file_name = fname
            self.parameters, data_offset = self.__parse_header(fname)

            AimArea = self.parameters['EDS Data']['AnalyzableMap MeasData']['Meas Cond']['Aim Area']
            if AimArea[1] == AimArea[3]:
                raise ValueError(f'"{fname}" does not contain map data! Aim area {AimArea} suggests to use `JEOL_DigiLine()` to load data.')

            if only_metadata:
                self.dcube = None
                self.drift_images = None
                self.frame_list = None
                self.nm_per_pixel = None
                self.__set_ref_spectrum()
                return

            self.frame_list = sorted(list(frame_list)) if split_frames and frame_list else None
            self.drift_images = self.__read_drift_images(fname, rebin) if read_drift else None
            self.dcube = self.__get_data_cube(dtype, data_offset,
                                              split_frames=split_frames,
                                              E_cutoff=E_cutoff,
                                              rebin=rebin,
                                              verbose=verbose)

            # Nominal pixel size [nm]
            ScanSize = self.parameters['PTTD Param']['Params']['PARAMPAGE0_SEM']['ScanSize']
            Mag = self.parameters['PTTD Data']['AnalyzableMap MeasData']['MeasCond']['Mag']
            self.nm_per_pixel = ScanSize / Mag * 1000000 / self.dcube.shape[2]

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
        fname : Str
            Filename.

        Returns
        -------
        header : Dict
            Dictionary containing all meta data stored in header.
        offset : Int
            Number of header bytes.

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

    def __get_data_cube(self, dtype, offset, split_frames=False,
                        E_cutoff=None, rebin=None, verbose=False):
        """Returns data cube (F x X x Y x E).

        Parameters
        ----------
        dtype : Str
            Data type used to store data cube in memory.
        offset : Int
            Number of header bytes.
        split_frames : Bool
            Store individual frames in the data cube (if True), otherwise sum
            all frames and store in a single frame (default).
        E_cutoff : Float
            Cutoff energy for spectra. Only store data below this energy.
        rebin : Tuple
            Rebin data while reading by (nv, nh). The integers nw and nh
            must be compatible with the scan size.
            None implied no rebinning performed.

        verbose : Bool
            Print additional output.

        Returns
        -------
        dcube : Ndarray (N x size x size x numCH)
            Data cube. N is the number of frames (if split_frames was selected)
            otherwise N=1, image is size x size pixels, spectra contain numCH
            channels.
        """
        # Verify that this is not DigiLine data
        AimArea = self.parameters['EDS Data'] \
                                 ['AnalyzableMap MeasData']['Meas Cond'] \
                                 ['Aim Area']
        assert AimArea[1] != AimArea[3]  # They are identical for DigiLine data

        CH_offset = self.__CH_offset_from_meta()
        NumCH = self.parameters['PTTD Param'] \
                               ['Params']['PARAMPAGE1_EDXRF'] \
                               ['NumCH']
        area = self. parameters['EDS Data'] \
                               ['AnalyzableMap MeasData']['Meas Cond'] \
                               ['Pixels'].split('x')

        if rebin is None:   # No rebinning required
            rebin = (1, 1)

        h = int(area[0]) // rebin[1]
        v = int(area[1]) // rebin[0]
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
        scale_h = 4096 / h
        scale_v = 4096 / v
        # map the size x size image into 4096x4096
        for d in data:
            N += 1
            if 32768 <= d < 36864:
                y = int((d - 32768) / scale_h)
            elif 36864 <= d < 40960:
                d = int((d - 36864) / scale_v)
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

    def __read_drift_images(self, fname, bs):
        """Read BF images stored in raw data

        Parameters
        ----------
        fname : Str
            Filename.
        bs: Tuple (nx, ny)
            Size of the bin applied, i.e. (2, 2) means that the output array will
            be reduced by a factor of 2 in both directions.


        Returns
        -------
        im : Ndarray or None
             Stack of images with shape (N_images, im_size, im_size) or None if
             no data is available.
             Rebinned data is returned if rebin is given.

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
        with open(fname, 'br') as f:
            f.seek(28)  # see self.__parse_header()
            data_pos = np.fromfile(f, '<I', 1)[0]
            f.seek(data_pos)
            rawdata = np.fromfile(f, dtype='u2')
            ipos = np.where(np.logical_and(rawdata >= 40960, rawdata < 45056))[0]
            if len(ipos) == 0:  # No data available
                return None

            im = np.array(rawdata[ipos] - 40960, dtype='uint16')
            try:
                return rebin(im.reshape(image_shape), bs)
            except ValueError:  # incomplete image
                # Add `N_addl` NaNs before reshape()
                N_addl = N_images * v * h - im.shape[0]
                im = np.append(im, np.full((N_addl), np.nan, dtype='uint16'))
                return rebin(im.reshape(image_shape), bs)

    def drift_statistics(self, filtered=False, verbose=False):
        """Returns 2D frequency distribution of frame shifts (x, y).

        Parameters
        ----------
        filtered : Bool
            If True, use Wiener filtered data.
        verbose : Bool
            Provide additional info if set to True.

        Returns
        -------
        h : Ndarray or None
            Histogram data or None if data cube contains a single frame only.
        extent : List
            Used to plot histogram as plt.imshow(h, extent=extent)

        Examples
        --------
        >>> from JEOL_eds import JEOL_pts
        >>> dc = JEOL_pts('data/128.pts', split_frames=True)

        Calculate the 2D frequency distribution of the frames shifts using unfiltered frames.
        >>> dc.drift_statistics()
        (array([[ 0.,  0.,  5.,  0.,  0.],
               [ 0.,  9.,  7.,  0.,  0.],
               [ 1., 10., 12.,  1.,  0.],
               [ 0.,  0.,  4.,  1.,  0.],
               [ 0.,  0.,  0.,  0.,  0.]]), [np.int64(-2), np.int64(2), np.int64(-2), np.int64(2)])

        Return the 2D frequency distribution of the Wiener filtered frames
        (plus extent useful for plotting).
        >>> m, e = dc.drift_statistics(filtered=True) #doctest: +NORMALIZE_WHITESPACE

        >>> import matplotlib.pyplot as plt
        >>> ax = plt.imshow(m, extent=e)
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
            extrema = np.where(h == np.amax(h))
            print(extrema)
            print('Shifts (filtered):') if filtered else print('Shifts (unfiltered):')
            if len(extrema) > 1:
                print('   Multiple maxima in 2D histogram of shifts detected!')
                for mx, my in zip(extrema[0], extrema[1]):
                    print(f'      {mx}, {my}')
                print(f'      Only considering {extrema[0][0]}, {extrema[1][0]}')
            mx = int(bins[int(extrema[0][0])] + 0.5)
            my = int(bins[int(extrema[1][0])] + 0.5)
            print(f'   Range: {int(np.asarray(sh).min())} - {int(np.asarray(sh).max())}')
            print(f'   Maximum {peak_val} at ({mx}, {my})')
        return h, extent

    def shifts(self, frames=None, filtered=False, verbose=False):
        """Calcultes frame shift by cross correlation of images (total intensity).

        Parameters
        ----------
        frames : Iterable
            Frame numbers for which shifts are calculated. First frame given is
            used a reference. Note, that the frame number denotes the index
            within the data cube loaded. This is different from the real frame
            number (stored in the `frame_list` attribute) if only a subset of
            frames was loaded.

        filtered : Bool
            If True, use Wiener filtered data.
        verbose : Bool
            Provide additional info if set to True.

        Returns
        -------
        shifts : List of tuples (dx, dy)
            List contains the shift for all frames or empty list if only a
            single frame is present.
            CAREFUL! Non-empty list ALWAYS contains 'meta.Sweeps' elements and
            contains (0, 0) for frames that were not in the list provided by
            keyword 'frames'.

        Examples
        --------
        >>> from JEOL_eds import JEOL_pts
        >>> dc = JEOL_pts('data/128.pts', split_frames=True)

        Get list of (possible) shifts [(dx0, dy0), (dx1, dx2), ...] in pixels
        of individual frames using frame 0 as reference. The shifts are
        calculated from the cross correlation of the images of the total x-ray
        intensity of each individual frame. Frame 0 used a reference.

        >>> sh = dc.shifts()
        >>> sh[-1]
        (0, -1)


        Use Wiener filtered images to calculate shifts:
        >>> sh = dc.shifts(filtered=True)
        >>> sh[-1]
        (0, -1)

        Calculate shifts for selected frames (odd frames) only. In this case
        farme 1, again the first frame given is used as reference
        >>> sh = dc.shifts(frames=range(1, dc.dcube.shape[0], 2))
        >>> sh[-1]
        (-1, -1)
        """
        if self.dcube is None or self.dcube.shape[0] == 1:
            # only a single frame present
            return []
        if frames is None:
            frames = range(self.dcube.shape[0])
        # Always use first frame given as reference
        ref = wiener(self.map(frames=[frames[0]])) if filtered else self.map(frames=[frames[0]])
        ref = ref.astype(float)
        ref -= ref.mean()
        shifts = [(0, 0)] * self.dcube.shape[0]
        if verbose:
            print(f'Frame {frames[0]} used a reference')
        for f in frames[1:]:    # skip reference frame
            data = wiener(self.map(frames=[f])) if filtered else self.map(frames=[f])
            data = data.astype(float)
            data -= data.mean()
            c = correlate(ref, data)
            # image size s=self.dcube.shape[1]
            # c has shape (2 * s - 1, 2 * s - 1)
            # Autocorrelation peaks at [s - 1, s - 1]
            # i.e. offset is at dy (dy) index_of_maximum - s + 1.
            dx, dy = np.where(c == np.amax(c))
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
        interval : Tuple (number, number)
            Defines interval (channels, or energy [keV]) to be used for map.
            None implies that all channels are integrated.
        energy : Bool
            If false (default) interval is specified as channel numbers
            otherwise (True) interval is specified as 'keV'.
        frames : Iterable (tuple, list, array, range object)
            Frame numbers included in map. If split_frames is active and frames
            is not specified all frames are included.
            Note, that the frame number denotes the index within the data cube
            loaded. This is different from the real frame number (stored in the
            `frame_list` attribute) if only a subset of frames was loaded.
        align : Str
            'no': Do not align individual frames.
            'yes': Align frames (use unfiltered frames in cross correlation).
            'filter': Align frames (use  Wiener filtered frames in cross correlation).
        verbose : Bool
            If True, output some additional info.

        Returns
        -------
        map : Ndarray
            Spectral Map.

        Examples
        --------
        >>> from JEOL_eds import JEOL_pts
        >>> import JEOL_eds.utils as JU

        >>> dc = JEOL_pts('data/128.pts', split_frames=True)

        Plot x-ray intensity integrated over all frames:
        >>> JU.plot_map(dc.map(), 'Greys_r')

        Only use given interval of energy channels to calculate map:
        >>> JU.plot_map(dc.map(interval=(115, 130)), 'Greys_r')

        Specify interval by energy [keV] instead of channel numbers:
        >>> JU.plot_map(dc.map(interval=(8,10), energy=True), 'Greys_r')

        If option 'split_frames' was used to read the data you can plot the map
        of a single frame:
        >>> JU.plot_map(dc.map(frames=[3]), 'inferno')

        Map corresponding to the sum of a few selected frames:
        >>> m = dc.map(frames=[3,5,11,12,13])

        Cu Kalpha map of all even frames:
        >>> m = dc.map(interval=(7.9, 8.1),
        ...            energy=True,
        ...            frames=range(0, dc.dcube.shape[0], 2))

        Correct for frame shifts (calculated from unfiltered frames):
        >>> m = dc.map(align='yes')
        >>> m.min()
        np.float64(0.0)
        >>> m.max()
        np.float64(137.0)

        Cu Kalpha map of frames 0..10. Frames are aligned using frame 5 as
        reference. Wiener filtered frames are used to calculate the shifts:
        >>> m = dc.map(interval=(7.9, 8.1),
        ...            energy=True,
        ...            frames=[5,0,1,2,3,4,6,7,8,9,10],
        ...            align='filter')
        >>> m.min()
        np.float64(0.0)
        >>> m.max()
        np.float64(5.0)
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
        shape = self.dcube.shape[1:3]     # image size
        if align == 'no':
            if frames is None:
                return self.dcube[:, :, :, interval[0]:interval[1]].sum(axis=(0, -1))
            # Only sum frames specified
            m = np.zeros(shape)
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
        Nx, Ny = shape
        res = np.zeros((2 * Nx, 2 * Ny))
        x0 = Nx // 2
        y0 = Nx // 2
        for f in frames:
            # map of this frame summed over all energy intervals
            dx, dy = shifts[f]
            res[x0 - dx:x0 - dx + Nx, y0 - dy:y0 - dy + Ny] += \
                self.dcube[f, :, :, interval[0]:interval[1]].sum(axis=-1)

        return res[x0:x0 + Nx, y0:y0 + Ny]

    def __spectrum_cROI(self, ROI, frames):
        """Returns spectrum integrated over a circular ROI

        Parameters
        ----------
        ROI : Tuple (center_x, center_y, radius)
        frames : Iterable (tuple, list, array, range object)
            Frame numbers included in spectrum. If split_frames is active and
            frames is not specified all frames are included.

        Returns
        -------
        spectrum : Ndarray
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
        ROI : Tuple (int, int)
            Tuple (int, int, int)
            Tuple (int, int, int, int)
            or None
            Defines ROI for which spectrum is extracted. None implies that
            the whole data cube is used.
            A tuple (v, h) defines a single point ROI given by its vertical and
            horizontal pixel index.
            A tuple (center_v, center_h, radius) defines a circular ROI
            including its boundary.
            A tuple (top, bottom, left, right) defines a rectangular ROI with
            boundaries included. Numbers are pixel indices in the range
            0 <= N < ImageSize. Note, that this definition implies y-axis
            before x-axis and the order of the numbers is the same as when
            applied in a python slice ([top:bottom, left:right]).
        frames : Iterable (tuple, list, array, range object)
            Frame numbers included in spectrum. If split_frames is active and
            frames is not specified all frames are included.
            Note, that the frame number denotes the index within the data cube
            loaded. This is different from the real frame number (stored in the
            `frame_list` attribute) if only a subset of frames was loaded.

        Returns
        -------
        spectrum :   Ndarray
            EDX spectrum

        Examples
        --------
        >>> from JEOL_eds import JEOL_pts
        >>> import JEOL_eds.utils as JU

        >>> dc = JEOL_pts('data/128.pts', split_frames=True)

        Plot spectrum integrated over full image. If option 'split_frames' was
        used to read the data the following plots spectra of all frames summed:
        >>> JU.plot_spectrum(dc.spectrum())

        The integrated spectrum is also stored in the raw data and can be
        accessed much quicker:
        >>> JU.plot_spectrum(dc.ref_spectrum)

        Plot spectrum corresponding to a single pixel. ROI is specified as
        tuple (v, h) of pixel coordinates:
        >>> JU.plot_spectrum(dc.spectrum(ROI=(45, 13)))

        Plot spectrum corresponding to a circular ROI specified as tuple
        (center_v, center_h, radius) of pixel coordinates:
        >>> JU.plot_spectrum(dc.spectrum(ROI=(80, 60, 15)))

        Plot spectrum corresponding to a (rectangular) ROI specified as tuple
        (top, bottom, left, right) of pixels:
        >>> JU.plot_spectrum(dc.spectrum(ROI=(10, 20, 50, 100)))

        Plot spectrum for a single frame ('split_frames' used):
        >>> JU.plot_spectrum(dc.spectrum(frames=[23]))

        Extract spectrum corresponding to a few frames summed:
        >>> spec = dc.spectrum(frames=[0,2,5,6])

        Spectrum of all odd frames summed:
        >>> spec = dc.spectrum(frames=range(1, dc.dcube.shape[0], 2))
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
            return _correct_spectrum(self.parameters,
                                     self.__spectrum_cROI(ROI, frames))

        # check that ROI lies fully within the data cube
        if not all(0 <= val < self.dcube.shape[1] for val in ROI):
            raise ValueError(f"ROI {ROI} lies partially outside data cube")

        if self.dcube.shape[0] == 1:   # only a single frame (0) present
            s = self.dcube[0, ROI[0]:ROI[1] + 1, ROI[2]:ROI[3] + 1, :].sum(axis=(0, 1))
            return _correct_spectrum(self.parameters, s)

        # split_frames is active
        if frames is None:  # no frames specified, sum all frames
            s = self.dcube[:, ROI[0]:ROI[1] + 1, ROI[2]:ROI[3] + 1, :].sum(axis=(0, 1, 2))
            return _correct_spectrum(self.parameters, s)

        # only sum specified frames
        s = np.zeros(self.dcube.shape[3], dtype=self.dcube.dtype)
        for frame in frames:
            s += self.dcube[frame, ROI[0]:ROI[1] + 1, ROI[2]:ROI[3] + 1, :].sum(axis=(0, 1))
        return _correct_spectrum(self.parameters, s)

    def time_series(self, interval=None, energy=False, frames=None):
        """Returns x-ray intensity integrated in `interval` for all frames.

        Parameters
        ----------
        interval : Tuple (number, number)
            Defines interval (channels, or energy [keV]) to be used for map.
            None implies that all channels are integrated.
        energy:   Bool
            If false (default) interval is specified as channel numbers
            otherwise (True) interval is specified as 'keV'.
        frames : Iterable (tuple, list, array, range object)
            Frame numbers included in time series (or None if all frames are
            used). The integrated number of counts is set to 'NaN' for all other
            frames. Note, that the frame number denotes the index within
            the data cube loaded. This is different from the real frame number
            (stored in the `frame_list` attribute) if only a subset of frames
            was loaded.

        Returns
        -------
            ts : Ndarray
                Time evolution of intergrated intensity in interval.

        Examples
        --------
        >>> from JEOL_eds import JEOL_pts

        >>> dc = JEOL_pts('data/128.pts', split_frames=True)

        Intgrate carbon Ka peak (interval specified as channels):
        >>> ts = dc.time_series(interval=(20,40))

        Integrate oxygen Ka peak (interval specified as energy [keV]) and remove
        a few bad frames (11,12) from time series:
        >>> frames = [f for f in range(dc.dcube.shape[0]) if f not in [11, 12]]
        >>> ts = dc.time_series(interval=(0.45, 0.6), energy=True, frames=frames)
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
                ts[f] = self.dcube[f, :, :, interval[0]:interval[1]].sum(axis=(0, 1, 2))
        return ts

    def make_movie(self, fname=None, only_drift=False, **kws):
        """Makes a movie of EDS data and drift_images

        Parameters
        ----------
        fname  : Str (or None)
            Filename for movie file. If none is supplied the base name of the
            '.pts' file is used.
        only_drift : Bool (False)
            If False (default), both drift image and EDX maps of the selected
            frames are shown next to each other in the movie. If set to True,
            only the drift images are shown. In this case ALL drift images are
            used even if only a subset of frames was loaded.

        Examples
        --------
        >>> from JEOL_eds import JEOL_pts

        >>> dc = JEOL_pts('data/128.pts', split_frames=True, read_drift=True)

        Make movie and store is as 'data/128.mp4':
        >>> dc.make_movie()

        Only use Cu K_alpha line:
        >>> dc.make_movie(interval=(7.9, 8.1), energy=True)

        Make movie (one frame only, drift_image will be blank) and save iT as
        'dummy.mp4':
        >>> dc.make_movie(fname='dummy.mp4')

        Only load a subset of frames (first two frames) but ALL drift images.
        >>> dc = JEOL_pts('data/128.pts', read_drift=True,
        ...               split_frames=True, frame_list=[0, 1])

        Only two frames have been loaded
        >>> dc.frame_list
        [0, 1]
        >>> dc.dcube.shape
        (2, 128, 128, 4000)

        All drift images have been loaded
        >>> dc.drift_images.shape
        (50, 128, 128)
        """
        if self.dcube is None:  # Only metadata was read
            return

        if only_drift and self.drift_images is None:
            # We have not loaded the drift images
            raise ValueError ("No drift images were loaded")

        if fname is None:
            fname = os.path.splitext(self.file_name)[0] + '.mp4'

        # remove `frames=` keyword from dict as it would interfere later
        try:
            kws.pop('frames')
        except KeyError:
            pass

        if only_drift:
            # Use all drift images
            frame_list = range(self.drift_images.shape[0])
        else:
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

        frames = []
        if only_drift:
            for i, STEM_image in enumerate(self.drift_images):
                frame = plt.imshow(STEM_image / STEM_max, animated=True)
                text = ax.annotate(i, (1, -5), annotation_clip=False)
                frames.append([frame, text])
        else:   # Use both, drift images and EDX maps
            # Note, more STEM images are present if only a subset of frames was
            # read.
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
        fname : Str (or None)
            Filename. If none is supplied the base name of the '.pts' file is used.

        Examples
        --------
        >>> from JEOL_eds import JEOL_pts

        >>> dc = JEOL_pts('data/128.pts', split_frames=True)

        Save extracted data cube. File name is the same as the '.pts' file but
        extension is changed to 'npz'. This makes only sense if option
        'split_frames' was NOT used to read the data. Otherwise the file size
        is MUCH larger than the '.pts' file and saving becomes time consuming:
        >>> dc.save_dcube()

        You can also supply your own filename, but use '.npz' as extension:
        >>> dc.save_dcube(fname='my_new_filename.npz')

        If you want to read the data cube into your own program:
        >>> npzfile = np.load('data/128.npz')
        >>> dcube = npzfile['arr_0']

        >>> dcube.shape
        (50, 128, 128, 4000)

        >>> dc.dcube.shape
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
        self.nm_per_pixel = None
        npzfile = np.load(fname)
        self.dcube = npzfile['arr_0']

    def save_hdf5(self, fname=None, **kws):
        """Saves all data including attributes to hdf5 file

        Parameters
        ----------
        fname : Str
            File name of '.h5' file (must end in '.h5'). If none is supplied the
            base name of the '.pts' file is used.

        Examples
        --------
        >>> from JEOL_eds import JEOL_pts

        >>> dc = JEOL_pts('data/128.pts', split_frames=True)

        Save data with file name based on the '.pts' file but extension is
        changed to 'h5':
        >>> dc.save_hdf5()

        You can also supply your own filename, but use '.h5' as extension:
        >>> dc.save_hdf5(fname='my_new_filename.h5')

        Pass along keyword arguments such as e.g. compression:
        >>> dc.save_hdf5('compressed.h5',
        ...              compression='gzip',
        ...              compression_opts=9)

        If you want to read the data cube into your own program:
        >>> hf = h5py.File('data/128.h5', 'r')

        List data sets available
        >>> for name in hf:
        ...     print(name)
        dcube

        Use '[()]' to get actual data not just a reference to stored data
        in file:
        >>> data = hf['dcube'][()]

        >>> data.shape
        (50, 128, 128, 4000)

        List attributes:
        >>> print(hf.attrs.keys())
        <KeysViewHDF5 ['file_date', 'file_name', 'nm_per_pixel', 'parameters']>

        >>> hf.attrs['file_name']
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
            hf.attrs['nm_per_pixel'] = self.nm_per_pixel
            if self.frame_list is not None:
                hf.attrs['frame_list'] = self.frame_list
            # avoid printing of ellipsis in arrays / lists
            np.set_printoptions(threshold=sys.maxsize)
            hf.attrs['parameters'] = str(self.parameters)

    def __load_hdf5(self, fname):
        """Loads data including attributes from hdf5 file

        Parameters
        ----------
        fname : Str
            File name of '.h5' file (must end in '.h5').
        """
        with h5py.File(fname, 'r') as hf:
            self.dcube = hf['dcube'][()]

            self.drift_images = hf['drift_images'][()] if 'drift_images' in hf.keys() else None
            self.frame_list = hf['drift_images'][()] if 'frame_list' in hf.keys() else None

            self.file_date = hf.attrs['file_date']
            self.file_name = hf.attrs['file_name']
            try:
                self.nm_per_pixel = hf.attrs['nm_per_pixel']
            except KeyError:
                self.nm_per_pixel = None

            aeval = asteval.Interpreter()
            self.parameters = aeval(hf.attrs['parameters'])


if __name__ == "__main__":
    import doctest
    doctest.testmod()
