#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 16:22:33 2021

@author: alxneit
"""
import unittest

from JEOL_eds import JEOL_pts
import numpy as np


class Metadata_pts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('Loading data ... ', end='', flush=True)
        cls.dc = JEOL_pts('data/128.pts', only_metadata=True)
        print('done')

    def test_read(self):
        self.assertEqual('data/128.pts', self.dc.file_name)
        self.assertEqual('2020-10-23 11:18:40', self.dc.file_date)
        self.assertIsNone(self.dc.nm_per_pixel)

        self.assertIsInstance(self.dc.parameters, dict)
        self.assertIsInstance(self.dc.ref_spectrum, np.ndarray)

        self.assertIsNone(self.dc.dcube)
        self.assertIsNone(self.dc.drift_images)

    def test_functions(self):
        self.assertIsNone(self.dc.spectrum())
        self.assertIsNone(self.dc.map())
        self.assertIsNone(self.dc.time_series())

        self.assertIsInstance(self.dc.shifts(), list)
        self.assertEqual([], self.dc.shifts())

        self.assertIsInstance(self.dc.drift_statistics(), tuple)
        self.assertEqual((None, None), self.dc.drift_statistics())

    def test_output(self):
        self.assertIsNone(self.dc.make_movie())
        self.assertIsNone(self.dc.save_dcube())
        self.assertIsNone(self.dc.save_hdf5())


if __name__ == '__main__':
    unittest.main()
