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
        (128, 128, 4096)

        # More info is stored in metadata
        >>>> dc.meta.N_ch    # Number of energy channels
        4096
        >>>> dc.meta.im_size # Map dimension (size x size)
        128
        # Print all metadata as dict
        >>>>: vars(dc.meta)
        {'N_ch': 4096,
         'CH_Res': 0.01,
         'im_size': 256,
         'E_calib': (0.0100006, -0.00122558),
         'LiveTime': 1638.1000000000001,
         'RealTime': 1692.6200000000001,
         'DwellTime': 0.5,
         'DeadTime': 'T4'}

        # Use helper functions map() and spectrum().
        >>>> import matplotlib.pyplot as plt

        # Use all energy channels, i.e. plot map of total number of counts
        >>>> plt.imshow(dc.map())
        <matplotlib.image.AxesImage at 0x7f7192ee6dd0>
        # Specify energy interval (channels containing a spectral line) to
        # be used for map. Used to map specific elements.
        >>>> plt.imshow(dc.map(interval=(115, 130)))
        <matplotlib.image.AxesImage at 0x7f7191eefd10>

        # Plot spectrum integrated over full dimension.
        >>>> plt.plot(dc.spectrum())
        [<matplotlib.lines.Line2D at 0x7f7192feec10>]
        # Plot spectrum corresponding to a (rectangular) ROI specified as
        # tuple (left, right, top, bottom) of pixels.
        >>>> plt.plot(dc.spectrum(ROI=(10,20,50,100)))
        [<matplotlib.lines.Line2D at 0x7f7192b58050>]

        # Save extracted data cube. File name is the same as the '.pts' file
        # but extension is changed to 'npz'.
        >>>> dc.save_dcube()

        # JEOL_pts object can also be initialized from a saved data cube. In
        # this case, dtype is the same as in the stored data cube and a
        # possible 'dtype=' keyword is ignored.
        >>>> dc2 = JEOL_pts('128.npz')
        >>>> dc2.file_name
        '128.npz'

        # If you want to read the data cube into your own program
        >>>> npzfile = np.load('128.npz')
        >>>> dcube = npzfile['arr_0']
        >>>> dcube.shape
        (128, 128, 4096)
    """

    def __init__(self, fname, dtype='uint16', debug=False):
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
        """
        if os.path.splitext(fname)[1] == '.npz':
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
        """Returns data cube (X x Y x E)

            Parameters
                dtype:      str
                            data type used to store data cube
                hsize:      int
                            number of header bytes
                Ndata:      int
                            number of data items ('u2') to be read

            Returns
                dcube:      numpy array (size x size x numCH)
                            data cube
        """
        with open(self.file_name, 'rb') as f:
            np.fromfile(f, dtype='u1', count=hsize)    # skip header
            data = np.fromfile(f, dtype='u2', count=Ndata)
        dcube = np.zeros([self.meta.im_size, self.meta.im_size, self.meta.N_ch], dtype=dtype)
        N = 0
        N_err = 0
        unknown = {}
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
                x = int((d - B) / scale)
            elif D <= d < E:
                z = int(d - D)
                dcube[x, y, z] = dcube[x, y, z] + 1
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

    def map(self, interval=None):
        """Returns map integrated over interval in spectrum

        Parameter
            interval:   tuple (int, int)
                        defines interval (channels) to be used for map.
                        None imples that all channels are integrated.


        Returns
            map:   ndarray
                   map
        """
        if not interval:
            interval = (0, self.meta.N_ch)

        return self.dcube[:, :, interval[0]:interval[1]].sum(axis=2)

    def spectrum(self, ROI=None):
        """Returns spectrum integrated over a ROI

        Parameter
                 ROI:   tuple (int, int, int, int) or None
                        defines ROI for which spectrum is extracted. ROI is
                        defined by its boundaries (left, right, top, bottom).
                        None implied that the whole image is used.
        Returns
            spectrum:   ndarray
                        spectrum
        """
        if not ROI:
            ROI = (0, self.meta.im_size, 0, self.meta.im_size)

        return self.dcube[ROI[0]:ROI[1], ROI[2]:ROI[3], :].sum(axis=0).sum(axis=0)

    def save_dcube(self):
        """Save (compressed) data cube as file_name.npz
        """
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
        self.meta.im_size = self.dcube.shape[0]
        self.meta.N_ch = self.dcube.shape[2]
