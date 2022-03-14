#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=C0103,C0301
"""
Copyright 2020-2021 Ivo Alxneit (ivo.alxneit@psi.ch)

This file is part of JEOL_eds.
JEOL_eds is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

JEOL_eds is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with JEOL_eds. If not, see <http://www.gnu.org/licenses/>.
"""

import os
import glob
import subprocess

# List of python files to run with doctest
files = glob.glob('JEOL_eds/JEOL_*')
files.append('JEOL_eds/utils.py')

# Output files to be deleted at end of test
output = ['128.h5',
          'carbon_Ka.pdf',
          'compressed.h5',
          'data/128.h5',
          'data/128.mp4',
          'data/128.npz',
          'dummy.mp4',
          'my_new_filename.h5',
          'my_new_filename.npz',
          'ref_spectrum.pdf',
          'test.pdf',
          'test_spectrum.dat',
          'test_tseries.dat']

# Run all doctests
for file in files:
    print('*' * 80)
    print('*', ' ' * 76, '*')
    N = 76 - len(file) - 13
    print(f'* doctesting {file} ', ' ' * N, '*')
    print('*', ' ' * 76, '*')
    print('*' * 80)
    cmd = f'{file} -v'
    subprocess.call(cmd, shell=True)
    print()
    print()

# Remove output

print('*' * 80)
print('*', ' ' * 76, '*')
N = 76 - 17
print('* removing output ', ' ' * N, '*')
print('*', ' ' * 76, '*')
print('*' * 80)
for file in output:
    print(file)
    os.remove(file)
