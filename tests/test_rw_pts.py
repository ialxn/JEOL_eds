#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 16:22:33 2021

@author: alxneit
"""
import unittest

from JEOL_eds import JEOL_pts
import numpy as np

class testIO_pts(unittest.TestCase):

    def test_read_basic(self):
        dc = JEOL_pts('data/128.pts')

        self.assertEqual((1, 128, 128, 4000), dc.dcube.shape)

        self.assertEqual('data/128.pts', dc.file_name)
        self.assertEqual('2020-10-23 11:18:40', dc.file_date)

        self.assertIsInstance(dc.parameters, dict)
        self.assertIsInstance(dc.ref_spectrum, np.ndarray)

        self.assertIsNone(dc.drift_images)

    def test_read_split_frames(self):
        dc = JEOL_pts('data/128.pts', split_frames=True)

        self.assertEqual((50, 128, 128, 4000), dc.dcube.shape)

        self.assertEqual('data/128.pts', dc.file_name)
        self.assertEqual('2020-10-23 11:18:40', dc.file_date)

        self.assertIsInstance(dc.parameters, dict)
        self.assertIsInstance(dc.ref_spectrum, np.ndarray)

        self.assertIsNone(dc.drift_images)

    def test_read_drift_images(self):
        dc = JEOL_pts('data/128.pts', read_drift=True)

        self.assertEqual((1, 128, 128, 4000), dc.dcube.shape)

        self.assertEqual('data/128.pts', dc.file_name)
        self.assertEqual('2020-10-23 11:18:40', dc.file_date)

        self.assertIsInstance(dc.parameters, dict)
        self.assertIsInstance(dc.ref_spectrum, np.ndarray)

        self.assertIsInstance(dc.drift_images, np.ndarray)
        self.assertEqual((50, 128, 128), dc.drift_images.shape)


if __name__ == '__main__':
    unittest.main()