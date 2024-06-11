#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 16:22:33 2021

@author: alxneit
"""
import unittest

from JEOL_eds import JEOL_pts


class subset_of_frames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        flist = [1, 2, 4, 8, 16]
        print('Loading data ... ', end='', flush=True)
        cls.dc = JEOL_pts('data/128.pts',
                          split_frames=True)
        cls.subset = JEOL_pts('data/128.pts',
                              split_frames=True, frame_list=flist)
        cls.flist = flist
        print('done')

    def test_dcube(self):
        self.assertEqual(self.subset.dcube.shape[0],
                         len(self.flist))
        for i, j in enumerate(self.flist):
            self.assertEqual(self.subset.dcube[i, :, :, :].sum(),
                             self.dc.dcube[j, :, :, :].sum())

    def test_individual_maps(self):
        for i, j in enumerate(self.flist):
            self.assertEqual(self.subset.map(frames=[i]).sum(),
                             self.dc.map(frames=[j]).sum())

    def test_sum_map(self):
        self.assertEqual(self.subset.map().sum(),
                         self.dc.map(frames=self.flist).sum())

    def test_individual_spectra(self):
        for i, j in enumerate(self.flist):
            self.assertEqual(self.subset.spectrum(frames=[i]).sum(),
                             self.dc.spectrum(frames=[j]).sum())

    def test_sum_spectrum(self):
        self.assertEqual(self.subset.spectrum().sum(),
                         self.dc.spectrum(frames=self.flist).sum())


if __name__ == '__main__':
    unittest.main()
