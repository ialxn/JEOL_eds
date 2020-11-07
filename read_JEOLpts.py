#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 13:30:08 2020

@author: alxneit
"""
import struct
import numpy as np

class JEOL_pts:
    """Work with JEOL '.pts' files

        In : dc = JEOL_pts('128.pts')
        2081741 902010

        In : dc.dcube.dtype
        Out: dtype('uint16')

        In : dc = JEOL_pts('128.pts', dtype='int')
        2081741 902010

        In : dc.dcube.dtype
        Out: dtype('int64')

        In : dc.file_name
        Out: '128.pts'

        In : dc.N_ch
        Out: 4096

        In : dc.im_size
        Out: 128

        In : dc.dcube.shape
        Out: (128, 128, 4096)

        In : plt.imshow(dc.map())
        Out: <matplotlib.image.AxesImage at 0x7f7192ee6dd0>

        In : plt.imshow(dc.map(interval=(115, 130)))
        Out: <matplotlib.image.AxesImage at 0x7f7191eefd10>

        In : plt.plot(dc.spectrum())
        Out: [<matplotlib.lines.Line2D at 0x7f7192feec10>]

        In : plt.plot(dc.spectrum(ROI=(10,20,50,100)))
        Out: [<matplotlib.lines.Line2D at 0x7f7192b58050>]

    """

    def __init__(self, fname, dtype='uint16'):
        """Read datacube from JEOL '.pts' file

            Parameters

                 fname:     str
                            filename
                 dtype:     str
                            data type used to store (not read) datacube.
                            can be any of the dtype supported by numpy.
        """
        self.file_name = fname
        headersize, datasize = self.__get_offset_and_size()
        self.im_size = self.__get_img_size(headersize)
        self.N_ch = self.__get_numCH(headersize)
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

    def __get_img_size(self, hsize):
        """Returns size of image.

            Parameters
                hsize:      int
                            number of header bytes

            Returns
                 size:      int
                            size of image (size x size) or None (Error)
        """
        sizes = [64, 128, 256, 512, 1024, 2048, 4092]   # possible image sizes
        with open(self.file_name, 'rb') as f:
            header = np.fromfile(f, dtype='u1', count=hsize)     # read header
            # search for string 'Pixels'
            for offset in range(hsize - 6):
                string = b''.join(list(struct.unpack('ssssss', header[offset:offset+6])))
                if string == b'Pixels':
                    # index to list is '<i2' 9 bytes after string (of length 6)
                    idx = np.frombuffer(header[offset+6+9: offset+6+11], dtype='<i2', count=1)[0]
                    return sizes[idx]
            return None

    def __get_numCH(self, hsize):
        """Returns number of channels (length of spectrum)

            Parameters
                hsize:      int
                            number of header bytes

            Returns
                numCH:      int
                            size of image (size x size) or None (Error)
        """
        with open(self.file_name, 'rb') as f:
            data = np.fromfile(f, dtype='u1', count=hsize)
            for offset in range(hsize - 5):
                string = b''.join(list(struct.unpack('sssss', data[offset:offset+5])))
                if string == b'NumCH':
                    numCH = np.frombuffer(data[offset+14: offset+14+4], dtype='<i4', count=1)[0]
                    return numCH
            return None


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
            data = np.fromfile(f, dtype='u2')
        dcube = np.zeros([self.im_size, self.im_size, self.N_ch], dtype=dtype)
        N = 0
        N_err = 0
        A = 2**15
        B = A + 4096
        C = B + 4096
        D = C + 4096
        E = D + self.N_ch
        # Data is mapped as follows:
        #   A <= datum < B   -> y-coordinate
        #   B <= datum < C   -> x-coordinate
        #   C + 4096 <= datum < C + 4096 + numCH    -> count registered
        scale = 4096 / self.im_size
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
                # I have no idea what these data mean
                N_err += 1
        print(N, N_err)
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
            interval = (0, self.N_ch)

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
            ROI = (0, self.im_size, 0, self.im_size)

        return self.dcube[ROI[0]:ROI[1], ROI[2]:ROI[3], :].sum(axis=0).sum(axis=0)
