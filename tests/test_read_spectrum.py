#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 11:17:05 2022

@author: alxneit
"""
import unittest

from JEOL_eds import JEOL_spectrum

class test_eds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('Loading data ... ', end='', flush=True)
        cls.s = JEOL_spectrum('data/spot.eds')
        print('done')

    def test_toplevel(self):
        self.assertEqual('data/spot.eds', self.s.file_name)
        self.assertEqual('2022-02-17 15:15:20', self.s.file_date)

    def test_spectral_data(self):
        self.assertEqual((4096,), self.s.data.shape)
        self.assertEqual(35860, self.s.data.sum())

    def test_header_data(self):
        self.assertEqual(4096, self.s.header['NumCH'])
        self.assertEqual(0.01, self.s.header['CH Res'])
        self.assertAlmostEqual(0.0100, self.s.header['CoefA'], 4)
        self.assertAlmostEqual(-0.00122558, self.s.header['CoefB'], 7)

    def test_footer_data(self):
        self.assertEqual(200.0, self.s.footer['Parameters']['AccKV'])
        self.assertEqual('JEM-ARM200F(HRP)', self.s.footer['Parameters']['SEM'])

if __name__ == '__main__':
    unittest.main()
