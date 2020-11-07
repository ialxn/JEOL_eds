#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:28:22 2020

@author: alxneit
"""
import struct
import numpy as np
import matplotlib.pyplot as plt


def get_offset_and_size(fname):
    """Returns length of header (bytes) and size of data (number of u2).

        Parameters
            fname:      str
                        filename

        Returns
            offset:     int
                        size of header (bytes) before data starts
            size:       int
                        number of data (u2) items
    """
    with open(fname, 'rb') as f:
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

def get_img_size(fname, hsize):
    """Returns size of image.

        Parameters
            fname:      str
                        filename
            hsize:      int
                        number of header bytes

        Returns
            size:       int
                        size of image (size x size) or None (Error)
    """
    sizes = [64, 128, 256, 512, 1024, 2048, 4092]   # possible image sizes
    with open(fname, 'rb') as f:
        header = np.fromfile(f, dtype='u1', count=hsize)     # read header
        # search for string 'Pixels'
        for offset in range(hsize - 6):
            string = b''.join(list(struct.unpack('ssssss', header[offset:offset+6])))
            if string == b'Pixels':
                # index to list is '<i2' 9 bytes after string (of length 6)
                idx = np.frombuffer(header[offset+6+9: offset+6+11], dtype='<i2', count=1)[0]
                return sizes[idx]
        return None

def get_numCH(fname, hsize):
    """Returns number of channels (length of spectrum)

        Parameters
            fname:      str
                        filename
            hsize:      int
                        number of header bytes

        Returns
            numCH:      int
                        size of image (size x size) or None (Error)
    """
    with open(fname, 'rb') as f:
        data = np.fromfile(f, dtype='u1', count=hsize)
        for offset in range(hsize - 5):
            string = b''.join(list(struct.unpack('sssss', data[offset:offset+5])))
            if string == b'NumCH':
                numCH = np.frombuffer(data[offset+14: offset+14+4], dtype='<i4', count=1)[0]
                return numCH
        return None


def get_data_cube(fname, hsize, Ndata, size, numCH):
    """Returns data cube (X x Y x E)

        Parameters
            fname:      str
                        filename
            hsize:      int
                        number of header bytes
            Ndata:      int
                        number of data items ('u2') to be read
             size:      int
                        image size (size x size)
            numCH:      int
                        number of energy channels (spectrum)

        Returns
            dcube:      numpy array (size x size x numCH)
                        data cube
    """
    with open(fname, 'rb') as f:
        np.fromfile(f, dtype='u1', count=hsize)    # skip header
        data = np.fromfile(f, dtype='u2')
    datacube = np.zeros([size, size, numCH])
    N = 0
    N_err = 0
    A = 2**15
    B = A + 4096
    C = B + 4096
    D = C + 4096
    E = D + numCH
    # Data is mapped as follows:
    #   A <= datum < B   -> y-coordinate
    #   B <= datum < C   -> x-coordinate
    #   C + 4096 <= datum < C + 4096 + numCH    -> count registered
    scale = 4096 / size
    # map the size x size image into 4096x4096
    for d in data:
        N += 1
        if A <= d < B:
            y = int((d - A) / scale)
        elif B <= d < C:
            x = int((d - B) / scale)
        elif D <= d < E:
            z = int(d - D)
            datacube[x, y, z] = datacube[x, y, z] + 1
        else:
            # I have no idea what these data mean
            N_err += 1
    print(N, N_err)
    return datacube

FN = '512.pts'
#FN = '256.pts'
#FN = '128.pts'
headersize, datasize = get_offset_and_size(FN)
size = get_img_size(FN, headersize)
N = get_numCH(FN, headersize)
dc = get_data_cube(FN, headersize, datasize, size, N)
#plt.imshow(dc.sum(axis=2)) # image
#plt.plot(dc.sum(axis=0).sum(axis=0)) # total spectrum
