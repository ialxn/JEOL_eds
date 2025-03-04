#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 16:22:33 2021

@author: alxneit
"""
import unittest

from JEOL_eds import JEOL_pts


class Spectrum(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print('\nLoading data ... ', end='', flush=True)
        cls.dc = JEOL_pts('data/128.pts', split_frames=True)
        print('done: ', end='', flush=True)

    def test_ref(self):
        # Skip lowest 100 channels where interpolation is performed
        self.assertEqual(self.dc.ref_spectrum[100:].sum(),
                         self.dc.spectrum()[100:].sum())

    def test_full_dcube(self):
        s = self.dc.spectrum()

        self.assertEqual((4000,), s.shape)
        self.assertEqual(s.sum(), 350308)

        # Skip lowest 100 channels where interpolation is performed
        self.assertEqual(s[100:].sum(),
                         self.dc.dcube[:, :, :, 100:].sum())

    def test_rectangular_ROI(self):
        ROI = (12, 34, 56, 78)
        s = self.dc.spectrum(ROI=ROI)

        self.assertEqual((4000,), s.shape)
        self.assertEqual(s.sum(), 16341)

        s2 = self.dc.dcube[:, ROI[0]:ROI[1] + 1, ROI[2]:ROI[3] + 1, :].sum(axis=(0, 1, 2))
        # Skip lowest 100 channels where interpolation is performed
        self.assertEqual(s[100:].sum(),
                         s2[100:].sum())

    def test_rectangular_ROI_frames(self):
        even = range(0, self.dc.dcube.shape[0], 2)
        odd = range(1, self.dc.dcube.shape[0], 2)

        ROI = (12, 34, 56, 78)
        s = self.dc.spectrum(ROI=ROI)

        s_even = self.dc.spectrum(ROI=ROI, frames=even)
        s_odd = self.dc.spectrum(ROI=ROI, frames=odd)
        # Skip lowest 100 channels where interpolation is performed
        s2_sum = s_even[100:].sum() + s_odd[100:].sum()
        self.assertAlmostEqual(s[100:].sum(), s2_sum, places=3)

    def test_circular_ROI(self):
        ROI = (65, 76, 41)
        s = self.dc.spectrum(ROI=ROI)

        self.assertEqual((4000,), s.shape)
        self.assertAlmostEqual(s.sum(), 193908.586, places=3)

    def test_circular_ROI_frames(self):
        even = range(0, self.dc.dcube.shape[0], 2)
        odd = range(1, self.dc.dcube.shape[0], 2)

        ROI = (65, 76, 41)
        s = self.dc.spectrum(ROI=ROI)

        s_even = self.dc.spectrum(ROI=ROI, frames=even)
        s_odd = self.dc.spectrum(ROI=ROI, frames=odd)
        s2_sum = s_even.sum() + s_odd.sum()
        self.assertAlmostEqual(s.sum(), s2_sum, places=3)

    def test_point_ROI(self):
        ROI = (78, 62)
        s = self.dc.spectrum(ROI=ROI)

        self.assertEqual((4000,), s.shape)
        self.assertAlmostEqual(s.sum(), 108.0, places=3)

    def test_point_ROI_frames(self):
        even = range(0, self.dc.dcube.shape[0], 2)
        odd = range(1, self.dc.dcube.shape[0], 2)

        ROI = (78, 62)
        s = self.dc.spectrum(ROI=ROI)

        s_even = self.dc.spectrum(ROI=ROI, frames=even)
        s_odd = self.dc.spectrum(ROI=ROI, frames=odd)
        # Skip lowest 100 channels where interpolation is performed
        s2_sum = s_even[100:].sum() + s_odd[100:].sum()
        self.assertAlmostEqual(s[100:].sum(), s2_sum, places=3)


if __name__ == '__main__':
    unittest.main()
