# JEOL_eds

A python package to read binary data files acquired by JEOL's **Analysis Station** software. The following data types are (at least partially) supported:

- EDS maps ('.pts'): Full area is acquired and stored.

- Scan line data ('.pts'): Data for a single (horizontal) scan line. Only the '.pts' file is used. Other data files present like '.dln' (profiles acquired for elements selected during data acquisition), '.img' (image used for positioning of the line), and '.map' (seems to be identical to '.img') can possibly be read separately (see below).

- Area and Spot scans ('.eds'): Contain full length EDS spectra.

- Line scans ('.pln'): Line scan data contain a given number of spots lying on an arbitrarily oriented line. The '.pln' file stores the list of these spots. Each corresponds to a separate spectrum stored in its '.eds' file found in the same directory.

- Image files ('.img', '.map')

The functions to parse the header of the binary '.pts', '.img', '.map', and '.eds' files were copied from HyperSpy (hyperspy/io_plugins/jeol.py scheduled for inclusion into HyperSpy 1.7).

This package does not aim to replace HyperSpy which is much more feature-rich. Instead it provides an easy interface to extract spectra or elemental maps from the binary file much like the *Play Back* feature in **Analysis Station**.



# Installation

## Requirements
```
Python 3.6+
numpy
scipy
matplotlib
scikit-image
asteval
h5py
(pip for installation)
```

Download zip and extract or clone repository. From the resulting folder run

```bash
$ pip install .
```
or

```bash
$ pip install . -U
```
to upgrade an existing installation.

# Important (API breaking) changes

## Version 2

- The parameter `frames=[]` lists the indices of a sub set of frames to be used. Since version 2 the indices refer to the original indices and not anymore to  their index in the data cube.
<br>
```
dc = JEOL_pts('data/128.pts', split_frames=True, frames=[10, 11, 12])
```
<br>
Now you refer to frame 11 (available as dc.dcube(1)) as e.g.:
<br>
```
m = dc.map(frames=[11])
```
<br>
while before:
<br>
```
m = dc.map(frames=[1])
```
<br>
A convenience function `JEOL_pts.frame()` is provide for easy access to the frames:
<br>
```
f = dc.frame(11)
```

- Data (EDX and drift images) can be rebinned on-the-fly while they are loaded. <br>
```
dc = JEOL_pts('data/128.pts', split_frames=True, rebin=(4, 4))
```

- The drift images can be loaded separately. The `read_drift=` parameter is now a string (`'yes'|'no'|'only'`) and not a boolean anymore.

- `JEOL_pts.make_movie(only_drift=True)` can now assemble a movie of the drift images only (independent of the presence of EDX data).

- The EDX frames can be (re)aligned based on the drift images (`align_src='data'|'drift_images'`).

# Usage

## General imports

```python
>>> from JEOL_eds import JEOL_pts, JEOL_spectrum, JEOL_image, JEOL_PointLine, JEOL_DigiLine
>>> import JEOL_eds.utils as JU
```


## EDS maps

```python
# Read binary EDS data (up to 11.0 keV) storing each sweep individually.
>>> dc = JEOL_pts('data/128.pts', split_frames=True, E_cutoff=11.0)

# Print calibration data (pixel size in nm) and size of data cube.
>>> dc.nm_per_pixel
1.93359375

>>> dc.dcube.shape
(50, 128, 128, 1100)

# If `split_frames=True` is used and the data cube becomes too big to be kept in
# memory a subset of frames can be read by using the keyword parameter `frame_list`.
>>> small_dc = JEOL_pts('data/128.pts',
                         split_frames=True, frame_list=[1,2,4,8,16],
                         E_cutoff=11.0)

# The frames in the data cube correspond to the original frames 1, 2, 4, 8, and 16.
>>> small_dc.frame_list
[1, 2, 4, 8, 16]

>>> small_dc.dcube.shape
(5, 128, 128, 1100)

# To read (and process) large data sets you might use the following code fragment.
# First read number of frames (read only meta data to speed up this step) and then
# process the full data set batch-wise.
>>> large_fn = 'data/128.pts'
>>> large = JEOL_pts(large_fn, only_metadata=True)
>>> N = large.parameters['EDS Data']['AnalyzableMap MeasData']['Doc']['Sweep']
>>> per_batch = 10
>>> N_batches = N // per_batch
>>> for i in range(N_batches):
        flist = [i*per_batch + j for j in range(per_batch)]
        subset = JEOL_pts(large_fn, split_frames=True, frame_list=flist)
        # Do the processing of the subset. Here print total X-ray intensity.
        print(subset.map().sum())
70627
70960
71021
70746
70777

# The fast way to read and plot reference spectrum.
>>> JU.plot_spectrum(JEOL_pts('data/64.pts', only_metadata=True).ref_spectrum)

# Cu Ka map of all even frames.
>>> dc = JEOL_pts('data/128.pts')
>>> m = dc.map(interval=(7.9, 8.1),
               energy=True,
               frames=range(0, dc.dcube.shape[0], 2))

# Cu Ka map of frames 0..10. Frames are aligned using frame 5 as
# reference. Wiener filtered frames are used to calculate the shifts.
>>> m = dc.map(interval=(7.9, 8.1),
               energy=True,
               frames=[5,0,1,2,3,4,6,7,8,9,10],
               align='filter')

# Plot nice map using custom color map black to purple. `gamma=0.9`
# enhances details. Little smoothing (FWHH=1.75 pixels) is applied.
>>> JU.plot_map(m, 'purple',
                label='Itot',
                background='black',
                gamma=0.9,
                smooth=0.75)

# Plot simple map that includes a scale bar.
>>> scale_bar = {'label': '50 nm',
                 'f_calib': dc.nm_per_pixel}
>>> JU.plot_map(m, 'purple',scale_bar=scale_bar)

# Plot rebinned (2x2) map to increase counts per pixel at decreased resolution.
>>> f_rebin = 2
>>> scale_bar = {'label': '50 nm',
                 'f_calib': dc.nm_per_pixel * f_rebin}
>>> JU.plot_map(JU.rebin(m, (f_rebin, f_rebin)),
                'purple',scale_bar=scale_bar)

# Overlays of elemental maps
>>> Fe = dc.map(interval=(6.25, 6.6), energy=True)
>>> Al = dc.map(interval=(1.4, 1.6), energy=True)
>>> Si = dc.map(interval=(1.65, 1.85), energy=True)

>>> scale_bar = {'label': '50 nm',
                 'f_calib': dc.nm_per_pixel}

>>> JU.create_overlay([Si, Al, Fe], ['blue', 'red', 'green'],
                      scale_bar=scale_bar)

```


## Scan line data
```python
>>> dl = JEOL_DigiLine('data/DigiLine/View000_0000003.pts')

# Report some meta data.
>>> dl.file_name
'data/DigiLine/View000_0000003.pts'

# Mag calibration factor.
>>> dl.nm_per_pixel
0.0099

# Data cube N x X x E (N_scans x N_pixels x N_E-channels).
>>> dl.dcube.shape
(50, 256, 4000)

# Full parameter set stored by Analysis Station is available via the
# `parameters` attribute. Here we query LiveTime.
>>> dl.parameters['PTTD Data']['AnalyzableMap MeasData']['Doc']['LiveTime']
63.13

# Plot part of reference spectrum (re-calibrated sum spectrum).
>>> JU.plot_spectrum(dl.ref_spectrum,
                     E_range=(1.2, 2.0),
                     M_ticks=(4,1))

# Plot sum spectrum of first 100 pixels in first scan.
>>> JU.plot_spectrum(dl.sum_spectrum(scans=[0], xRange=(0, 100)))

# Extract oxygen profile, x-axis [nm].
>>> x, p_O = dl.profile(interval=(0.45, 0.6),
                        energy=True, xCalib=True)

# Spectral map (spectrum versus position) of energies up to 2.5 keV.
>>> m = dl.spectral_map(E_range=(0, 2.5), energy=True)
>>> m.shape
(256, 250)

>>> JU.plot_map(m, 'red')
```


## Area or spot scan data
```python
>>> s = JEOL_spectrum('data/spot.eds')

# Size of spectral data.
>>> s.data.shape
(4096,)

# Report some meta data.
>>> s.file_name
'data/spot.eds'

# Display some meta data of header.
>>> header = s.header
>>> header['CountRate']
1238.0

# Display some meta data of footer.
>>> footer = s.footer
>>> footer['Parameters']['AccKV']
200.0

# Plot data.
>>> JU.plot_spectrum(s.data,
                     E_range=(0, 20),
                     M_ticks=(4, 1))
```


## Line scan data
```python
>>> pl = JEOL_PointLine('data/PointLine/View000_0000001.pln')

# Report some meta data. '.pln' file contains list of spectra and image.
>>> pl.file_name
'View000_0000001.pln'

>>> pl.Image_name.
'View000_0000000.img'

# The attribute `eds_dict` is a dict with `marker` as key and a list with
# [FileName, xPos, yPos] as content.
>>> pl.eds_dict
{0: ['View000_0000006.eds', 85.3125, 96.4375],
 1: ['View000_0000005.eds', 81.4375, 92.6875],
 2: ['View000_0000004.eds', 77.5625, 88.9375],
 3: ['View000_0000003.eds', 73.6875, 85.1875],
 4: ['View000_0000002.eds', 69.8125, 81.4375]}

# Image object (`JEOL_image`) is stored as attribute `ref_image`.
>>> ref = pl.ref_image

>>> ref.file_name.
'data/PointLine/View000_0000000.img'

# Image parameters can be accessed such as MAG calibration.
>>> ref.nm_per_pixel
1.93359375

# Spectral data is available.
>>> pl.eds_data.shape
(5, 4096)

# Visualize the position of the individual data points on a zoomed-in image.
>>> pl.show_PointLine(ROI=(45,110,50,100),
                      color='red',
                      ann_color='white')

# Extract profile of Ti Ka line with one spectrum (marker '2') omitted
# x axis in [nm].
>>> x, p_Ti = pl.profile(interval=(4.4, 4.65),
                      energy=True,
                      markers=[0, 1, 3, 4],
                      xCalib=True)
>>> x
array([ 0.        , 10.42673734, 20.85347467, 31.28021201, 41.70694934])

>>> JU.plot_profile(x, p_Ti, units='nm')
```


## Image files
```python
# Read an image file.
>>> demo_im = JEOL_image('data/demo.img')

>>> demo_im.file_name
'data/demo.img'

# Report some meta data stored in file.
>>> demo_im.parameters['Instrument']['Name']
'JEM-ARM200F(HRP)'
>>> demo_im.parameters['Image']['Size']
array([512, 512], dtype=int32)

# Read a map file.
>>> demo_map = JEOL_image('data/demo.map')

# Print calibration data (pixel size in nm).
>>> demo_map.nm_per_pixel
3.8671875

# Plot image with scale bar.
# 'data/demo.img' is a BF image. Thus use inverted color map.
>>> scale_bar = {'label': '200nm',
                 'f_calib': demo_map.nm_per_pixel,
                 'color': 'black'}
>>> JU.plot_map(demo_map.image, 'inferno_r', scale_bar=scale_bar)
```


# Bugs

## Possibly conflicting parameter types

Parameters loaded from '.pts' files might have different types than the ones
loaded from '.h5' files. Thus take extra care if you need to compare them:

```python
# Load and store as hdf5.
>>> dc = JEOL_pts('data/128.pts')
>>> dc.save_hdf5(compression='gzip', compression_opts=9)
# Initialize from hdf5
>>> dc_hdf5 = JEOL_pts('data/128.h5')

# Compare parameters dict gives unexpected result.
>>> p = dc.parameters['PTTD Data']['AnalyzableMap MeasData']['MeasCond']
>>> p_hdf5 = dc_hdf5.parameters['PTTD Data']['AnalyzableMap MeasData']['MeasCond']
>>> p == p_hdf5
False

# But they seem identical.
>>> p
{'AccKV': 200.0,
 'AccNA': 7.475,
 'Mag': 800000,
 'WD': 3.2,
 'ScanR': 270.0,
 'FocusMP': 16043213}

>>> p_hdf5
{'AccKV': 200.0,
 'AccNA': 7.475,
 'Mag': 800000,
 'WD': 3.2,
 'ScanR': 270.0,
 'FocusMP': 16043213}

# The issue is different types.
# This works.
>>> p['AccKV'] == p_hdf5['AccKV']
True
>>> type(p['AccKV'])
numpy.float32
>>> type(p_hdf5['AccKV'])
float

# This causes the issue.
>>> p['AccNA'] == p_hdf5['AccNA']
False
>>> type(p['AccNA'])
numpy.float32
>>> type(p_hdf5['AccNA'])
float
```

## Missing position information

The position in the '.img' file where **spot**, **area** ('.eds.') data were acquired is still unknown (they must be present in the data sets but I have not yet been able to extract them).

