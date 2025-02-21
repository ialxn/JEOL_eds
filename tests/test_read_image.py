#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 16:22:33 2021

@author: alxneit
"""
import unittest

from JEOL_eds import JEOL_image

class test_img(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('\nLoading data ... ', end='', flush=True)
        cls.demo = JEOL_image('data/demo.img')
        print('done: ', end='', flush=True)

    def test_toplevel(self):
        self.assertEqual('data/demo.img', self.demo.file_name)
        self.assertEqual('2021-08-13 16:09:06', self.demo.file_date)
        self.assertEqual('JED-2200:LibJxImageForm', self.demo.fileformat)
        self.assertEqual(1.93359375, self.demo.nm_per_pixel)

    def test_imagedata(self):
        self.assertEqual((512, 512), self.demo.image.shape)
        self.assertTrue((self.demo.image == self.demo.parameters["Image"]["Bits"]).all())
        self.assertEqual(63782847, self.demo.image.sum())

class test_map(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('\nLoading data ... ', end='', flush=True)
        cls.demo = JEOL_image('data/demo.map')
        print('done: ', end='', flush=True)

    def test_toplevel(self):
        self.assertEqual('data/demo.map', self.demo.file_name)
        self.assertEqual('2021-08-13 16:09:36', self.demo.file_date)
        self.assertEqual('JED-2200:LibJxImageForm', self.demo.fileformat)
        self.assertEqual(3.8671875, self.demo.nm_per_pixel)

    def test_imagedata(self):
        self.assertEqual((256, 256), self.demo.image.shape)
        self.assertTrue((self.demo.image == self.demo.parameters["Image"]["Bits"]).all())
        self.assertEqual(116726, self.demo.image.sum())

if __name__ == '__main__':
    unittest.main()
