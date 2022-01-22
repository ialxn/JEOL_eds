#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 16:22:33 2021

@author: alxneit
"""
import unittest
import os

from JEOL_eds import JEOL_pts
import numpy as np

class testIO_hdf5(unittest.TestCase):

    def tearDown(self):
        os.remove('data/test.h5')

    def test_rw_hdf5(self):
        dc = JEOL_pts('data/64.pts', split_frames=True, read_drift=True)
        dc.save_hdf5('data/test.h5', compression='gzip', compression_opts=9)
        saved = JEOL_pts('data/test.h5')

        self.assertEqual(dc.file_name, saved.file_name)
        self.assertEqual(dc.file_date, saved.file_date)
        self.assertEqual(dc.nm_per_pixel, saved.nm_per_pixel)

        self.assertEqual(dc.dcube.shape, saved.dcube.shape)
        self.assertEqual(dc.dcube.sum(), saved.dcube.sum())

        self.assertEqual(dc.drift_images.shape, saved.drift_images.shape)
        self.assertEqual(dc.drift_images.sum(), saved.drift_images.sum())

        self.assertIsInstance(saved.parameters, dict)

        self.assertIsInstance(saved.ref_spectrum, np.ndarray)
        self.assertEqual(dc.ref_spectrum.sum(), saved.ref_spectrum.sum())

        self.assertEqual(dc.parameters['EDS Data']['AnalyzableMap MeasData']['MeasCond']['FocusMP'],
                         saved.parameters['EDS Data']['AnalyzableMap MeasData']['MeasCond']['FocusMP'])


if __name__ == '__main__':
    unittest.main()