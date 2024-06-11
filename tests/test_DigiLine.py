#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 11:17:05 2022

@author: alxneit
"""
import unittest

from JEOL_eds import JEOL_DigiLine


class test_DigiLine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('Loading data ... ', end='', flush=True)
        cls.dl = JEOL_DigiLine('data/DigiLine/View000_0000003.pts')
        print('done')

    def test_toplevel(self):
        self.assertEqual('data/DigiLine/View000_0000003.pts', self.dl.file_name)
        self.assertEqual('2022-03-04 13:19:43', self.dl.file_date)
        self.assertEqual(0.1546875, self.dl.nm_per_pixel)

    def test_parameters1(self):
        p = self.dl.parameters['EDS Data']['AnalyzableMap MeasData']
        self.assertEqual(63.13, p['Doc']['LiveTime'])
        self.assertEqual('256x256', p['Meas Cond']['Pixels'])

    def test_parameters2(self):
        data = self.dl.parameters['EDS Data']['AnalyzableMap MeasData']['Data']['EDXRF']
        self.assertEqual((4096,), data.shape)
        self.assertEqual(30759, data.sum())

    def test_Aim(self):
        p = self.dl.parameters['EDS Data']['AnalyzableMap MeasData']
        aim = p['Meas Cond']['Aim Area']
        self.assertEqual(0, aim[0])
        self.assertEqual(144, aim[1])
        self.assertEqual(255, aim[2])
        self.assertEqual(aim[1], aim[3])

    def test_ref_spectrum(self):
        data = self.dl.ref_spectrum
        self.assertEqual((4096,), data.shape)
        self.assertEqual(30759, data.sum())

    def test_data_cube(self):
        data = self.dl.dcube
        self.assertEqual((50, 256, 4000), data.shape)
        self.assertEqual(31178, data.sum())

    def test_profile1(self):
        x, p = self.dl.profile()
        self.assertEqual((256,), p.shape)
        self.assertEqual(x.shape, p.shape)
        self.assertEqual(32640.0, x.sum())
        self.assertEqual(31178, p.sum())

    def test_profile2(self):
        x, p = self.dl.profile(interval=(1.4, 1.6), energy=True, xCalib=True)
        self.assertEqual((256,), p.shape)
        self.assertEqual(x.shape, p.shape)
        self.assertEqual(5049.0, x.sum())
        self.assertEqual(6740, p.sum())

    def test_sum_spectrum(self):
        spectrum = self.dl.sum_spectrum()
        self.assertEqual((4000,), spectrum.shape)
        self.assertEqual(30710, spectrum.sum())
        scans = range(1, 50, 3)
        spectrum = self.dl.sum_spectrum(xRange=(24, 123), scans=scans)
        self.assertEqual((4000,), spectrum.shape)
        self.assertEqual(4304, spectrum.sum())

    def test_spectral_map(self):
        m = self.dl.spectral_map(E_range=(0, 2.5), energy=True)
        self.assertEqual((256, 250,), m.shape)
        self.assertEqual(26539, m.sum())


if __name__ == '__main__':
    unittest.main()
