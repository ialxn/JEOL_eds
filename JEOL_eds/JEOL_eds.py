#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 13:30:08 2020

@author: alxneit
"""
import os
import struct
import numpy as np

class EDS_metadata:
    """Class to contain metadata
    """
    def __init__(self, header):
        """Populates meta data from parameters stored in header

            Parameter
                header:     byte array (or None)
                            Binary header or None if we were loading
                            a '.npz' file that does not contain metadata.
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
                ParName:    str
                            Name of parameter to be extracted
                header:     byte array
                            Binary header

            Returns
                value:      Any type (depends on parameter extracted)
                            Value of parameter. Single number or array
                            (or None, if reading from '.npz' file, i.e.
                             when header is None)

            Notes
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

        # Initialize JEOL_pts object (read data from '.pts' file).
        # Data cube has dtype = 'uint16' (default).
        >>>> dc = JEOL_pts('128.pts')
        >>>> dc.dcube.dtype
        dtype('uint16')

        # Same, but specify dtype to be used for data cube.
        >>>> dc = JEOL_pts('128.pts', dtype='int')
        >>>> dc.dcube.dtype
        dtype('int64')

        # Provide some debug output when loading.
        >>>> dc = JEOL_pts('128.pts', dtype='uint16', debug=True)
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

        # Store individual frames
        >>>> dc=JEOL_pts('test/128.pts', split_frames=True)
        >>>> dc.dcube.shape
        (50, 128, 128, 4096)

        # More info is stored in metadata
        >>>> dc.meta.N_ch    # Number of energy channels
        4096
        >>>> dc.meta.im_size # Map dimension (size x size)
        128
        # Print all metadata as dict
        >>>>: vars(dc.meta)
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
        # If split_frames is active the following plots maps for all frames
        # added together.
        >>>> plt.imshow(dc.map())
        <matplotlib.image.AxesImage at 0x7f7192ee6dd0>
        # Specify energy interval (channels containing a spectral line) to
        # be used for map. Used to map specific elements.
        >>>> plt.imshow(dc.map(interval=(115, 130)))
        <matplotlib.image.AxesImage at 0x7f7191eefd10>
        # specify interval by energy (keV) instead of channel numbers.
        >>>>plt.imshow(p_off.map(interval=(8,10), units=True))
        <matplotlib.image.AxesImage at 0x7f4fd0616950>
        # If split_frames is active you can specify to plot the map
        # of a single frame
        >>>> plt.imshow(dc.map(frames=(3)))
        <matplotlib.image.AxesImage at 0x7f06c05ef750>
        # Map correponding to a few frames
        >>>> m = dc.map(frames=(3,5,11,12,13))
        # Cu Kalpha map of all even frames
        >>>> m = dc.map(interval=(7.9, 8.1),
                        energy=True,
                        frames=range(0, dc.meta.Sweep, 2))

        # Plot spectrum integrated over full dimension. If split_frames is
        # active the following plots spectra for all frames added together.
        >>>> plt.plot(dc.spectrum())
        [<matplotlib.lines.Line2D at 0x7f7192feec10>]
        # Plot spectrum corresponding to a (rectangular) ROI specified as
        # tuple (left, right, top, bottom) of pixels.
        >>>> plt.plot(dc.spectrum(ROI=(10,20,50,100)))
        [<matplotlib.lines.Line2D at 0x7f7192b58050>]
        # Plot spectrum for a single frame (if split_frames is active).
        >>>> plt.plot(dc.spectrum(frames=(23)))
        [<matplotlib.lines.Line2D at 0x7f06b3db32d0>]
        # Extract spectrum corresponding to a few frames added
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

        # If you want to read the data cube into your own program
        >>>> npzfile = np.load('128.npz')
        >>>> dcube = npzfile['arr_0']
        # Single frame or split_frames was not active when data was saved
        >>>> dcube.shape
        (1, 128, 128, 4096)
        # Split_frames was active when data was saved
        >>>> dcube.shape
        (50, 128, 128, 4096)
    """

    def __init__(self, fname, dtype='uint16', debug=False, split_frames=False):
        """Read datacube from JEOL '.pts' file or from previously saved data cube

            Parameters

                 fname:     str
                            filename
                 dtype:     str
                            data type used to store (not read) datacube.
                            can be any of the dtype supported by numpy.
                            if a '.npz' file is loaded, this parameter is
                            ignored and the dtype corresponds to the one
                            of the loaded data cube.
                 debug:     bool
                            Turn on (various) debug output.
          split_frames:     bool
                            store individual frames in the data cube (if
                            True), otherwise add all frames and store in
                            a single frame (default).
        """
        self.split_frames = split_frames
        if os.path.splitext(fname)[1] == '.npz':
            self.debug = None
            self.meta = EDS_metadata(None)
            self.__load_dcube(fname)
        else:
            self.file_name = fname
            self.debug = debug
            headersize, datasize = self.__get_offset_and_size()
            with open(fname, 'rb') as f:
                header = np.fromfile(f, dtype='u1', count=headersize)
                self.meta = EDS_metadata(header)
            self.dcube = self.__get_data_cube(dtype, headersize, datasize)


    def __get_offset_and_size(self):
        """Returns length of header (bytes) and size of data (number of u2).

            Returns
                offset:     int
                            size of header (bytes) before data starts
                  size:     int
                            number of data (u2) items
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

    def __get_data_cube(self, dtype, hsize, Ndata):
        """Returns data cube (F x X x Y x E)

            Parameters
                dtype:      str
                            data type used to store data cube
                hsize:      int
                            number of header bytes
                Ndata:      int
                            number of data items ('u2') to be read

            Returns
                dcube:      numpy array (N x size x size x numCH)
                            data cube. N is the number of frames (if
                            split_frames was selected) otherwise N=1.
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
        A = 2**15
        B = A + 4096
        C = B + 4096
        D = C + 4096
        E = D + self.meta.N_ch
        # Data is mapped as follows:
        #   A <= datum < B   -> y-coordinate
        #   B <= datum < C   -> x-coordinate
        #   C + 4096 <= datum < C + 4096 + numCH    -> count registered
        scale = 4096 / self.meta.im_size
        # map the size x size image into 4096x4096
        for d in data:
            N += 1
            if A <= d < B:
                y = int((d - A) / scale)
            elif B <= d < C:
                d = int((d - B) / scale)
                if self.split_frames and d < x:
                    # A new frame starts once the slow axis (x) restarts which
                    # is not necessary at zero, if we have very few counts and
                    # nothing registers on scan line x=0.
                    frame += 1
                x = d
            elif D <= d < E:
                z = int(d - D)
                dcube[frame, x, y, z] = dcube[frame, x, y, z] + 1
            else:
                if self.debug:
                    # I have no idea what these data mean
                    # collect statistics on these values for debug
                    if str(d) in unknown:
                        unknown[str(d)] += 1
                    else:
                        unknown[str(d)] = 1
                    N_err += 1
        if self.debug:
            print('Unidentified data items ({} out of {}, {:.2f}%) found:'.format(N, N_err, 100*N_err/N))
            for key in sorted(unknown):
                print('\t{}: found {} times'.format(key, unknown[key]))
        return dcube

    def map(self, interval=None, energy=False, frames=None):
        """Returns map integrated over interval in spectrum

        Parameter
            interval:   tuple (number, number)
                        defines interval (channels, or energy [keV]) to be used
                        for map.
                        None implies that all channels are integrated.
              energy:   bool
                        If false (default) interval is specified as channel
                        numbers otherwise (True) interval is specified as 'keV'.
              frames:   iterable (tuple, list, array, range object)
                        Frame numbers included in map. If split_frames is
                        active and frames is not specified all frames are
                        included.

        Returns
            map:   ndarray
                   map
        """
        if not interval:
            interval = (0, self.meta.N_ch)
        if energy:
            interval = (int(round((interval[0] - self.meta.E_calib[1]) / self.meta.E_calib[0])),
                        int(round((interval[1] - self.meta.E_calib[1]) / self.meta.E_calib[0])))
        if self.debug:
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
        """Returns spectrum integrated over a ROI

        Parameter
                 ROI:   tuple (int, int, int, int) or None
                        defines ROI for which spectrum is extracted. ROI is
                        defined by its boundaries (left, right, top, bottom).
                        None implied that the whole image is used.
              frames:   iterable (tuple, list, array, range object)
                        Frame numbers included in spectrum. If split_frames is
                        active and frames is not specified all frames are included.

        Returns
            spectrum:   ndarray
                        spectrum
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
        """Save (compressed) data cube

            Parameter
                fname:  str (or None)
                        filename. If none is supplied the basename
                        of the '.pts' file is used.
        """
        if fname is None:
            fname = os.path.splitext(self.file_name)[0] + '.npz'
        np.savez_compressed(fname, self.dcube)

    def __load_dcube(self, fname):
        """Initialize by loading from previously saved data cube

        Parameter
            fname:  str
                    file name of '.npz' file (must end in '.npz')
        """
        self.file_name = fname
        npzfile = np.load(fname)
        self.dcube = npzfile['arr_0']
        self.meta.Sweep = self.dcube.shape[0]
        if self.meta.Sweep > 1:
            self.split_frames = True
        self.meta.im_size = self.dcube.shape[1]
        self.meta.N_ch = self.dcube.shape[3]
