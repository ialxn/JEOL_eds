#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 13:30:08 2020

@author: alxneit
"""
import os
import struct
import numpy as np
from scipy.signal import wiener, correlate

class EDS_metadata:
    """Class to store metadata.
    """
    def __init__(self, header):
        """Populates meta data from parameters stored in header

            Parameters
            ----------
                header:     Byte array (or None)
                            Binary header or None if loading a '.npz'
                            file that does not contain metadata.
        """
        self.N_ch = self.__get_parameter('NumCH', header)
        self.CH_Res = self.__get_parameter('CH Res', header)
        try:
            self.Sweep = self.__get_parameter('Sweep', header[1024:])
        except TypeError:
            pass
        self.im_size = self.__get_parameter('ScanLine', header)
        self.E_calib = (self.__get_parameter('CoefA', header),
                        self.__get_parameter('CoefB', header))
        self.LiveTime = self.__get_parameter('LiveTime', header)
        self.RealTime = self.__get_parameter('RealTime', header)
        self.DwellTime = self.__get_parameter('DwellTime(msec)', header)
        try:
            self.DeadTime = 'T' + str(self.__get_parameter('DeadTime', header) + 1)
        except TypeError:
            self.DeadTime = None

    @staticmethod
    def __get_parameter(ParName, header):
        """Returns parameter value extracted from header (or None).

            Parameters
            ----------
                ParName:    Str
                            Name of parameter to be extracted.
                 header:    Byte array
                            Binary header.

            Returns
            -------
                value:      Any type (depends on parameter extracted)
                            Value of parameter. Single number or array
                            (or None, if reading from '.npz' file, i.e.
                            when header is None).

            Notes
            -----
                According to jeol_metadata.ods. Right after the parameter name
                the numerical type of the parameter is encoded as '<u4' followed
                by how many bytes are used to store it (parameter might contain
                multiple items). This table is (partially) stored in a dict.

                    3: ('<u4', 4)       code 3 -> 'uint32' is 4 bytes long
        """
        if header is None:
            return None
        items = {2 : ('<u2', 2),
                 3 : ('<u4', 4),
                 4 : ('<f4', 4),
                 5 : ('<f8', 8),
                 8 : ('<f4', 4)     # list of float32 values
                 }
        ParLen = len(ParName)
        FormStr = b's' * ParLen
        for offset in range(header.size - ParLen):
            string = b''.join(list(struct.unpack(FormStr, header[offset:offset+ParLen])))
            if string == ParName.encode():
                offset += ParLen + 1
                # After parameter name, first read code
                ItemCode = np.frombuffer(header[offset: offset+4], dtype='<i4', count=1)[0]
                offset += 4
                # Read number of bytes needed to store parameter
                NBytes = np.frombuffer(header[offset: offset+4], dtype='<i4', count=1)[0]
                NItems = int(NBytes / items[ItemCode][1])   # How many items to be read
                offset += 4
                val = np.frombuffer(header[offset: offset+NBytes],
                                    dtype=items[ItemCode][0],
                                    count=NItems)
                if val.size == 1:   # return single number
                    return val[0]
                return val          # return array
        return None

class JEOL_pts:
    """Work with JEOL '.pts' files

        Examples
        --------

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
        Unidentified data items (2081741 out of 902010, 43.33%) found:
	        24576: found 41858 times
	        28672: found 40952 times
	        40960: found 55190 times
               .
               .
               .
	        41056: found 1 times
	        41057: found 1 times
	        41058: found 1 times

        # Useful attributes
        >>>> dc.file_name
        '128.pts'               # File name loaded from.
        >>>> dc.dcube.shape     # Shape of data cube
        (1, 128, 128, 4096)

        # Store individual frames.
        >>>> dc=JEOL_pts('test/128.pts', split_frames=True)
        >>>> dc.dcube.shape
        (50, 128, 128, 4096)

        # More info is stored in metadata.
        >>>> dc.meta.N_ch    # Number of energy channels
        4096
        >>>> dc.meta.im_size # Map dimension (size x size)
        128
        # Print all metadata as dict.
        >>>> vars(dc.meta)
        {'N_ch': 4096,
         'CH_Res': 0.01,
         'Sweep': 50,
         'im_size': 256,
         'E_calib': (0.0100006, -0.00122558),
         'LiveTime': 1638.1000000000001,
         'RealTime': 1692.6200000000001,
         'DwellTime': 0.5,
         'DeadTime': 'T4'}

        # Use helper functions map() and spectrum().
        >>>> import matplotlib.pyplot as plt

        # Use all energy channels, i.e. plot map of total number of counts.
        # If split_frames is active, the following draws maps for all frames
        # added together.
        >>>> plt.imshow(dc.map())
        <matplotlib.image.AxesImage at 0x7f7192ee6dd0>
        # Specify energy interval (channels containing a spectral line) to
        # be used for map. Used to map specific elements.
        >>>> plt.imshow(dc.map(interval=(115, 130)))
        <matplotlib.image.AxesImage at 0x7f7191eefd10>
        # Specify interval by energy (keV) instead of channel numbers.
        >>>>plt.imshow(p_off.map(interval=(8,10), units=True))
        <matplotlib.image.AxesImage at 0x7f4fd0616950>
        # If split_frames is active you can specify to plot the map
        # of a single frame
        >>>> plt.imshow(dc.map(frames=(3)))
        <matplotlib.image.AxesImage at 0x7f06c05ef750>
        # Map correponding to a few frames.
        >>>> m = dc.map(frames=(3,5,11,12,13))
        # Cu Kalpha map of all even frames
        >>>> m = dc.map(interval=(7.9, 8.1),
                        energy=True,
                        frames=range(0, dc.meta.Sweep, 2))

        # Plot spectrum integrated over full image. If split_frames is
        # active the following plots spectra for all frames added together.
        >>>> plt.plot(dc.spectrum())
        [<matplotlib.lines.Line2D at 0x7f7192feec10>]
        # Plot spectrum corresponding to a (rectangular) ROI specified as
        # tuple (left, right, top, bottom) of pixels.
        >>>> plt.plot(dc.spectrum(ROI=(10,20,50,100)))
        <matplotlib.lines.Line2D at 0x7f7192b58050>
        # Plot spectrum for a single frame (if split_frames is active).
        >>>> plt.plot(dc.spectrum(frames=(23)))
        <matplotlib.lines.Line2D at 0x7f06b3db32d0>
        # Extract spectrum corresponding to a few frames added.
        >>>> spec = dc.spectrum(frames=(0,2,5,6))
        # Spectrum of all odd frames
        >>>> spec = dc.spectrum(frames=range(1, dc.meta.sweep, 2))

        # Save extracted data cube. File name is the same as the '.pts' file
        # but extension is changed to 'npz'.
        >>>> dc.save_dcube()
        # You can also supply your own filename, but use '.npz' as extension.
        >>>> dc.save_dcube(fname='my_new_filename.npz')

        # JEOL_pts object can also be initialized from a saved data cube. In
        # this case, dtype is the same as in the stored data cube and a
        # possible 'dtype=' keyword is ignored.
        >>>> dc2 = JEOL_pts('128.npz')
        >>>> dc2.file_name
        '128.npz'

        # Get list of (possible) shifts [(dx0, dy0), (dx1, dx2), ...] in pixels
        # of individual frames using frame 0 as reference. The shifts are
        # calculated from the cross correlation of images of total intensity of
        # each individual frame.
        # Set "filtered=True" to calculate shifts from Wiener filtered images.
        >>>> dc.shifts()
        [(0, 0),
         (array([1]), array([1])),
         (array([0]), array([0])),
         .
         .
         .
         (array([1]), array([2]))]
        >>>> dc.shifts(filtered=True)
        /.../miniconda3/lib/python3.7/site-packages/scipy/signal/signaltools.py:1475: RuntimeWarning: divide by zero encountered in true_divide
         res *= (1 - noise / lVar)
        /.../miniconda3/lib/python3.7/site-packages/scipy/signal/signaltools.py:1475: RuntimeWarning: invalid value encountered in multiply
         res *= (1 - noise / lVar)
        [(0, 0),
         (array([1]), array([1])),
         (array([0]), array([1])),
         .
         .
         .
         (array([1]), array([2]))]

        # If you want to read the data cube into your own program.
        >>>> npzfile = np.load('128.npz')
        >>>> dcube = npzfile['arr_0']
        # split_frames was not active when data was saved.
        >>>> dcube.shape
        (1, 128, 128, 4096)
        # Split_frames was active when data was saved.
        >>>> dcube.shape
        (50, 128, 128, 4096)
    """

    def __init__(self, fname, dtype='uint16', verbose=False, split_frames=False):
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
               verbose:     Bool
                            Turn on (various) output.
          split_frames:     Bool
                            Store individual frames in the data cube (if
                            True), otherwise add all frames and store in
                            a single frame (default).
        """
        self.split_frames = split_frames
        if os.path.splitext(fname)[1] == '.npz':
            self.meta = EDS_metadata(None)
            self.__load_dcube(fname)
        else:
            self.file_name = fname
            headersize, datasize = self.__get_offset_and_size()
            with open(fname, 'rb') as f:
                header = np.fromfile(f, dtype='u1', count=headersize)
                self.meta = EDS_metadata(header)
            self.dcube = self.__get_data_cube(dtype, headersize, datasize,
                                              verbose=verbose)


    def __get_offset_and_size(self):
        """Returns length of header (bytes) and size of data (number of u2).

            Returns
            -------
                offset:     Int
                            Size of header (bytes) before data starts.
                  size:     Int
                            Number of data (u2) items.
        """
        with open(self.file_name, 'rb') as f:
            np.fromfile(f, dtype='u1', count=4)     # skip
            data = np.fromfile(f, dtype='u1', count=8)  # magic string
            ftype = b''.join(list(struct.unpack('ssssssss', data)))
            if ftype != b'PTTDFILE':    # wrong file format
                raise ValueError('Wrong file format')
            np.fromfile(f, dtype='u1', count=16)    # skip
            offset = np.fromfile(f, dtype='<i4', count=1)[0]
            size = np.fromfile(f, dtype='<i4', count=1)[0]  # total length bytes
            size = (size - offset) / 2  # convert to number of u2 in data segment
            return offset, int(size)

    def __get_data_cube(self, dtype, hsize, Ndata, verbose=False):
        """Returns data cube (F x X x Y x E).

            Parameters
            ----------
                dtype:      Str
                            Data type used to store data cube in memory.
                hsize:      Int
                            Number of header bytes.
                Ndata:      Int
                            Number of data items ('u2') to be read.
              verbose:      Bool
                            Print additional output

            Returns
            -------
                dcube:      Ndarray (N x size x size x numCH)
                            Data cube. N is the number of frames (if split_frames
                            was selected) otherwise N=1, image is size x size pixels,
                            spectra contain numCH channels.
        """
        with open(self.file_name, 'rb') as f:
            np.fromfile(f, dtype='u1', count=hsize)    # skip header
            data = np.fromfile(f, dtype='u2', count=Ndata)
        if self.split_frames:
            dcube = np.zeros([self.meta.Sweep, self.meta.im_size, self.meta.im_size, self.meta.N_ch], dtype=dtype)
        else:
            dcube = np.zeros([1, self.meta.im_size, self.meta.im_size, self.meta.N_ch], dtype=dtype)
        N = 0
        N_err = 0
        unknown = {}
        frame = 0
        x = -1
        # Data is mapped as follows:
        #   32768 <= datum < 36864                  -> y-coordinate
        #   36864 <= datum < 40960                  -> x-coordinate
        #   45056 <= datum < END (=45056 + N_ch)    -> count registered
        END = 45056 + self.meta.N_ch
        scale = 4096 / self.meta.im_size
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
                ##################################################
                #                                                #
                #  tentative OFFSET by 96 channels (see #59_60)  #
                #                                                #
                ##################################################
                z -= 96
                if z >= 0:
                    dcube[frame, x, y, z] = dcube[frame, x, y, z] + 1
            else:
                if verbose:
                    # I have no idea what these data mean
                    # collect statistics on these values for debug
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

    def shifts(self, filtered=False):
        """Calcultes frame shift by cross correlation of images (total intensity).

            Parameters
            ----------
             filtered:     Bool
                           If True, use Wiener filtered data.

            Returns
            -------
                            List of tuples (dx, dy) containing the shift for
                            all frames or empty list if only a single frame
                            is present.
        """
        if self.meta.Sweep == 1:
            # only a single frame present
            return []
        # Use first frame as reference
        if filtered:
            ref = wiener(self.map(frames=[0]))
        else:
            ref = self.map(frames=[0])
        shifts = [(0, 0)]
        for f in range(1, self.meta.Sweep):
            if filtered:
                c = correlate(ref, wiener(self.map(frames=[f])))
            else:
                c = correlate(ref, self.map(frames=[f]))
            dx, dy = np.where(c==np.amax(c))
            # FIX: More than one maximum is possible
            shifts.append((self.meta.im_size - dx, self.meta.im_size - dy))
        return shifts

    def map(self, interval=None, energy=False, frames=None, verbose=False):
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
                 verbose:   Bool
                            If True, output some additional info.

            Returns
            -------
                map:   Ndarray
                       Spectral Map.
        """
        if not interval:
            interval = (0, self.meta.N_ch)
        if energy:
            interval = (int(round((interval[0] - self.meta.E_calib[1]) / self.meta.E_calib[0])),
                        int(round((interval[1] - self.meta.E_calib[1]) / self.meta.E_calib[0])))
        if verbose:
            print('Using channels {} - {}'.format(interval[0], interval[1]))

        if not self.split_frames:   # only a single frame (0) present
            return self.dcube[0, :, :, interval[0]:interval[1]].sum(axis=-1)

        # split_frame is active
        if frames is None:  # no frames specified, sum all frames
            return self.dcube[:, :, :, interval[0]:interval[1]].sum(axis=(0, -1))

        # only sum specified frames
        m = np.zeros((self.dcube.shape[1:3]))
        for frame in frames:
            m += self.dcube[frame, :, :, interval[0]:interval[1]].sum(axis=-1)
        return m

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
            ROI = (0, self.meta.im_size, 0, self.meta.im_size)
        if not self.split_frames:   # only a single frame (0) present
            return self.dcube[0, ROI[0]:ROI[1], ROI[2]:ROI[3], :].sum(axis=(0, 1))

        # split_frames is active
        if frames is None:  # no frames specified, sum all frames
            return self.dcube[:, ROI[0]:ROI[1], ROI[2]:ROI[3], :].sum(axis=(0, 1, 2))

        # only sum specified frames
        spec = np.zeros(self.dcube.shape[-1])
        for frame in frames:
            spec += self.dcube[frame, ROI[0]:ROI[1], ROI[2]:ROI[3], :].sum(axis=(0, 1))
        return spec

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
        npzfile = np.load(fname)
        self.dcube = npzfile['arr_0']
        self.meta.Sweep = self.dcube.shape[0]
        if self.meta.Sweep > 1:
            self.split_frames = True
        self.meta.im_size = self.dcube.shape[1]
        self.meta.N_ch = self.dcube.shape[3]
