#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 16:22:33 2021

@author: alxneit
"""
import unittest
import warnings

from JEOL_eds import JEOL_pts


class Map(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('\nLoading data ... ', end='', flush=True)
        cls.dc = JEOL_pts('data/128.pts', split_frames=True)
        print('done: ', end='', flush=True)

    def test_full_dcube(self):
        m = self.dc.map()

        self.assertEqual((128, 128), m.shape)
        self.assertEqual(m.sum(), 354131)
        self.assertEqual(m.sum(), self.dc.dcube.sum())

    def test_full_dcube_frames(self):
        even = range(0, self.dc.dcube.shape[0], 2)
        odd = range(1, self.dc.dcube.shape[0], 2)

        m_even = self.dc.map(frames=even)
        m_odd = self.dc.map(frames=odd)
        m = self.dc.map()
        self.assertEqual(m.sum(), m_even.sum() + m_odd.sum())

    def test_interval_CH(self):
        CH = (123, 456)
        m = self.dc.map(interval=CH)
        m2 = self.dc.dcube[:, :, :, CH[0]:CH[1]].sum(axis=(0, -1))
        self.assertEqual(m.sum(), m2.sum())

    def test_interval_CH_frames(self):
        even = range(0, self.dc.dcube.shape[0], 2)
        odd = range(1, self.dc.dcube.shape[0], 2)
        CH = (123, 456)

        m_even = self.dc.map(interval=CH, frames=even)
        m_odd = self.dc.map(interval=CH, frames=odd)
        m = self.dc.map(interval=CH)
        self.assertEqual(m.sum(), m_even.sum() + m_odd.sum())

    def test_interval_E(self):
        E = (1.23, 4.56)
        CH = (int(round(E[0] * 100)), int(round(E[1] * 100)))

        m = self.dc.map(interval=E, energy=True)
        m2 = self.dc.dcube[:, :, :, CH[0]:CH[1]].sum(axis=(0, -1))
        self.assertEqual(m.sum(), m2.sum())

    def test_interval_E_frames(self):
        even = range(0, self.dc.dcube.shape[0], 2)
        odd = range(1, self.dc.dcube.shape[0], 2)
        E = (1.23, 4.56)

        m_even = self.dc.map(interval=E, energy=True, frames=even)
        m_odd = self.dc.map(interval=E, energy=True, frames=odd)
        m = self.dc.map(interval=E, energy=True)

        self.assertEqual(m.sum(), m_even.sum() + m_odd.sum())

    def test_align_yes(self):
        warnings.filterwarnings("ignore")
        m = self.dc.map(align='yes')

        self.assertEqual((128, 128), m.shape)
        self.assertEqual(m.sum(), 352658.0)

    def test_align_yes_frames(self):
        pass

    def test_align_filter(self):
        warnings.filterwarnings("ignore")
        m = self.dc.map(align='filter')

        self.assertEqual((128, 128), m.shape)
        self.assertEqual(m.sum(), 352881.0)

    def test_align_filter_frames(self):
        pass


if __name__ == '__main__':
    unittest.main()
