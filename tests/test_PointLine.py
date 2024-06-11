#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 11:17:05 2022

@author: alxneit
"""
import unittest

import numpy as np

from JEOL_eds import JEOL_PointLine


class test_PointLine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('Loading data ... ', end='', flush=True)
        cls.pl = JEOL_PointLine('data/PointLine/View000_0000001.pln')
        print('done')

    def test_toplevel(self):
        self.assertEqual('View000_0000001.pln', self.pl.file_name)
        self.assertEqual('View000_0000000.img', self.pl.Image_name)
        self.assertEqual('LG10004 ; 41.676 nm', self.pl.eds_header['sp_name'])

    def test_eds_list(self):
        self.assertEqual(5, len(self.pl.eds_dict))
        self.assertEqual('View000_0000006.eds', self.pl.eds_dict[0][0])
        self.assertAlmostEqual(81.4375, self.pl.eds_dict[1][1], 6)
        self.assertAlmostEqual(88.9375, self.pl.eds_dict[2][2], 6)

    def test_ref_image(self):
        self.assertEqual('data/PointLine/View000_0000000.img', self.pl.ref_image.file_name)
        self.assertEqual('2022-02-17 15:21:48', self.pl.ref_image.file_date)
        self.assertAlmostEqual(1.93359375, self.pl.ref_image.nm_per_pixel, 7)
        self.assertEqual('JED-2200:IMG', self.pl.ref_image.parameters['FileType'])
        self.assertEqual((256, 256), self.pl.ref_image.image.shape)
        self.assertEqual(2643253, self.pl.ref_image.image.sum())

    def test_eds_data(self):
        self.assertEqual((5, 4096), self.pl.eds_data.shape)
        self.assertEqual(25090.0, self.pl.eds_data.sum())

    def test_profile_1(self):
        x, p = self.pl.profile()
        self.assertEqual((5,), p.shape)
        self.assertEqual(x.shape, p.shape)
        self.assertEqual(6046.0, p[0])
        self.assertEqual(0.0, x[0])
        self.assertEqual(p.sum(), self.pl.eds_data.sum())
        self.assertAlmostEqual(53.924136, x.sum(), 5)

    def test_profile_2(self):
        x, p = self.pl.profile(interval=(4.4, 4.65), energy=True)
        self.assertEqual((5,), p.shape)
        self.assertEqual(x.shape, p.shape)
        self.assertEqual(1765.0, p[0])
        self.assertEqual(7123.0, p.sum())

    def test_profile_3(self):
        x, p = self.pl.profile(interval=(4.4, 4.65), energy=True)
        x, pa = self.pl.profile(interval=(4.4, 4.65), energy=True, markers=[0, 1])
        x, pb = self.pl.profile(interval=(4.4, 4.65), energy=True, markers=[2, 3, 4])
        self.assertEqual((5,), pa.shape)
        self.assertEqual((5,), pb.shape)
        self.assertEqual(4194.0, np.nansum(pa))
        self.assertEqual(2929.0, np.nansum(pb))
        self.assertEqual(4194.0, pa[0:2].sum())
        self.assertEqual(2929.0, pb[2:].sum())
        self.assertEqual(0.0, np.nansum(pa) + np.nansum(pb) - p.sum())

    def test_profile_4(self):
        x, p = self.pl.profile(xCalib=True)
        self.assertEqual((5,), p.shape)
        self.assertEqual(x.shape, p.shape)
        self.assertEqual(6046.0, p[0])
        self.assertEqual(0.0, x[0])
        self.assertEqual(p.sum(), self.pl.eds_data.sum())
        self.assertAlmostEqual(104.26737, x.sum(), 5)


if __name__ == '__main__':
    unittest.main()
