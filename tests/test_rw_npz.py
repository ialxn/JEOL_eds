#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 16:22:33 2021

@author: alxneit
"""
import unittest
import os

from JEOL_eds import JEOL_pts

class testIO_npz(unittest.TestCase):

    def tearDown(self):
        os.remove('data/test.npz')

    def test_rw_npz(self):
        dc = JEOL_pts('data/64.pts')
        dc.save_dcube('data/test.npz')
        saved = JEOL_pts('data/test.npz')

        self.assertEqual('data/test.npz', saved.file_name)
        self.assertIsNone(saved.file_date)
        self.assertIsNone(saved.drift_images)
        self.assertIsNone(saved.parameters)

        self.assertEqual(dc.dcube.shape, saved.dcube.shape)
        self.assertEqual(dc.dcube.sum(), saved.dcube.sum())


if __name__ == '__main__':
    unittest.main()