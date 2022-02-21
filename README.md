# JEOL_eds

A python module to read binary data files ('.pts'), image files ('.img', '.map'), and spectral data ('.eds', 'pln') acquired by JEOL's Analysis Station software. The functions to parse the header of the binary file and of the '.img', '.map', and '.eds' files were copied from HyperSpy (hyperspy/io_plugins/jeol.py scheduled for inclusion into HyperSpy 1.7).

This module does not aim to replace HyperSpy which is much more feature-rich. Instead it provides an easy interface to extract spectra or elemental maps from the binary file much like the *Play Back* feature in **Analysis Station**.



## Installation

### Requirements
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

## Usage
```python
>>>> from JEOL_eds import JEOL_pts, JEOL_spectrum, JEOL_image, JEOL_PointLine
>>>> import JEOL_eds.utils as JU

# Read binary EDS data storing each sweep individually.
>>>> dc = JEOL_pts('data/128.pts', split_frames=True, E_cutoff=11.0)

# If `split_frames=True` is used and the data cube becomes too big to be kept in
# memory a subset of frames can be read by using the keyword parameter `list_frames`.
>>>> small_dc = JEOL_pts('data/128.pts',
                         split_frames=True, list_frames=[1,2,4,8,16],
                         E_cutoff=11.0)
>>>> small_dc.frame_list
[1, 2, 4, 8, 16]

>>>> small_dc.dcube.shape
(5, 128, 128, 1100)
# The frames in the data cube correspond to the original frames 1, 2, 4, 8, and 16.

# Read and plot image data.
>>>> demo = JEOL_image('data/demo.img')
>>>> JU.plot_map(demo.image, 'Greys_r')

# Add scle bar. 'data/demo.img' is a BF image. Thus use inverted color map.
>>>> scale_bar = {'label': '200nm',
                  'f_calib': demo.nm_per_pixel,
                  'color': 'white'}
>>>> plot_map(demo.image, 'inferno_r', scale_bar=scale_bar)

# Report meta data of image file.
>>>> demo.parameters
{'Instrument': {'Type': 0,
  'ScanSize': 198.0,
  'Name': 'JEM-ARM200F(HRP)',
  'AccV': 200.0,
  'Currnnt': 7.475,
  'Mag': 200000,
  'WorkD': 3.2,
  'ScanR': 0.0},
 'FileType': 'JED-2200:IMG',
 'Image': {'Created': 44421.67298611111,
  'GroupName': '',
  'Memo': '',
  'DataType': 1,
  'Size': array([512, 512], dtype=int32),
  'Bits': array([[255, 255, 255, ..., 255, 255, 255],
         [255, 255, 255, ..., 255, 255, 255],
         [255, 255, 255, ..., 255, 255, 255],
         ...,
         [255, 255, 255, ..., 255, 255, 255],
         [255, 255, 255, ..., 255, 255, 255],
         [255, 255, 255, ..., 255, 255, 255]], dtype=uint8),
  'Title': 'IMG1'},
 'Palette': {'RGBQUAD': array([       0,    65793,   131586,   197379,   263172,   328965,
           394758,   460551,   526344,   592137,   657930,   723723,
           789516,   855309,   921102,   986895,  1052688,  1118481,
          ...,
         16185078, 16250871, 16316664, 16382457, 16448250, 16514043,
         16579836, 16645629, 16711422, 16777215], dtype=int32),
  '4': {'0': {'Pos': 0, 'Color': 0}, '1': {'Pos': 255, 'Color': 16777215}},
  'Active': 1,
  'Min': 0.0,
  'Max': 255.0,
  'Contrast': 1.0,
  'Brightness': -0.0,
  'Scheme': 1}}

# Read a map file.
>>>> demo = JEOL_image('data/demo.map')

# Print calibration data (pixel size in nm).
>>>> demo.pixel_size
0.99

# To read (and process) large data sets you might use the following code fragment.
# First read number of frames (read only meta data to speed up this step) and then
# process the full data set batch-wise.
>>>> large_fn = 'data/128.pts'
>>>> large = JEOL_pts(large_fn, only_metadata=True)
>>>> N = large.parameters['EDS Data']['AnalyzableMap MeasData']['Doc']['Sweep']
>>>> per_batch = 10
>>>> N_batches = N // per_batch
>>>> for i in range(N_batches):
         flist = [i*per_batch + j for j in range(per_batch)]
         subset = JEOL_pts(large_fn, split_frames=True, frame_list=flist)
         # Do the processing of the subset. Here print total X-ray intensity.
         print(subset.map().sum())

# Extract Cu Kalpha map of all even frames.
>>>> m = dc.map(interval=(7.9, 8.1),
                energy=True,
                frames=range(0, dc.dcube.shape[0], 2))

# Cu Kalpha map of frames 0..10. Frames are aligned using frame 5 as reference.
# Wiener filtered frames are used to calculate the shifts.
# Verbose output.
>>>> m = dc.map(interval=(7.9, 8.1),
                energy=True,
                frames=[5,0,1,2,3,4,6,7,8,9,10],
                align='filter',
                verbose=True)
Using channels 790 - 810
Frame 5 used a reference
/../scipy/signal/signaltools.py:1475: RuntimeWarning: divide by zero encountered in true_divide
  res *= (1 - noise / lVar)
/../scipy/signal/signaltools.py:1475: RuntimeWarning: invalid value encountered in multiply
  res *= (1 - noise / lVar)

# Plot spectrum integrated over full image. If option `split_frames` was used to
# read the data the following plots the sum spectrum of all frames added.
>>>> JU.plot_spectrum(dc.spectrum())

# The sum spectrum of the whole data cube is also stored in the raw data and can
# be accessed much faster.
>>>> JU.plot_spectrum(dc.ref_spectrum)

# Plot sum spectrum corresponding to a (rectangular) ROI specified as tuple
# (top, bottom, left, light) of pixels for selected frames.
# Verify definition of ROI before you apply it using the total X-ray intensity
# as image.
>>>> ROI = (10, 20, 50, 100)
>>>> JU.show_ROI(dc.map(), ROI, alpha=0.6)
>>>> JU.plot_spectrum(dc.spectrum(ROI=ROI, frames=[0,1,2,10,11,12,30,31,32]))

# Create overlay of elemental maps.
# Load data from a HDF5 fie. Data does not contain drift images and all frames
# were added, thus only a single frame is present.
>>>> dc = JEOL_pts('data/complex_oxide.h5')

# Extract some elemental maps. Where possible, add contribution of several lines.
>>>> Ti = dc.map(interval=(4.4, 5.1), energy=True)      # Ka,b
>>>> Fe = dc.map(interval=(6.25, 6.6), energy=True)     # Ka
>>>> Sr = dc.map(interval=(13.9, 14.4), energy=True)    # Ka
>>>> Co = dc.map(interval=(6.75, 7.0), energy=True)     # Ka
>>>> Co += dc.map(interval=(7.5, 7.8), energy=True)     # Kb
>>>> O = dc.map(interval=(0.45, 0.6), energy=True)

# Visualize the CoFe distribution using first of the `drift_images` as background.
# NOTE: Drift images were not stored in the data supplied and this will raise
#       TypeError: 'NoneType' object is not subscriptable
>>>> JU.create_overlay([Fe, Co],
                       ['Maroon', 'Violet'],
                       legends=['Fe', 'Co'],
                       BG_image=dc.drift_images[0])

# Plot and save reference spectrum between 1.0 and 2.5 keV. Plot one minor tick
# on x-axis and four on y-axis. Pass additional keywords to
# `matplotlib.pyplot.plot()`.
>>>> JU.plot_spectrum(dc.ref_spectrum,
                      E_range=(4, 17.5),
                      M_ticks=(1, 4),
                      outfile='ref_spectrum.pdf',
                      color='Red', linestyle='-.', linewidth=1.0)

# To insert the output of the different plot functions imported from
# `JEOL_eds.utils` into a sub-plot, use the following code fragment:

>>>> import matplotlib.pyplot as plt
>>>> fig, (ax1, ax2) = plt.subplots(1, 2)
# Use `ax1` for overlay
>>>> plt.sca(ax1)
>>>> JU.create_overlay((O, Sr, Ti),
                       ('Blue', 'Green', 'Red'),
                       legends=['O', 'Sr', 'Ti'],
                       BG_image=dc.drift_images[0])
# Use `ax2` for spectrum
>>>> plt.sca(ax2)
>>>> JU.plot_spectrum(dc.ref_spectrum, E_range=(4, 17.5))
>>>> plt.tight_layout() 	# Prevents overlapping labels
>>>> plt.savefig('demo.pdf')

# Calculate and plot line profiles.
# Extract carbon map
>>>> C_map = dc.map(interval=(0.22, 0.34), energy=True)

# Define line. Verify definition.
>>>> line = (80, 5, 110, 100)
>>>> width = 10
>>>> JU.show_line(C_map, line, linewidth=width, cmap='inferno')

# Calculate profile along a given line (width equals 10 pixels) and plot it.
>>>> profile = get_profile(C_map, line, linewidth=width)
>>>> import matplotlib.pyplot as plt
>>>> plt.plot(profile)

# Make movie of drift_images and total EDS intensity and store it
# as 'data/128.mp4'.
>>>> dc = JEOL_pts('data/128.pts', split_frames=True, read_drift=True)
>>>> dc.make_movie()

# Check for contamination by carbon. Integrate carbon Ka line.
>>>> ts = dc.time_series(interval=(0.45, 0.6), energy=True)

# Plot the time series and save output.
>>>> JU.plot_tseries(ts,
                     M_ticks=(9,4),
                     outfile='carbon_Ka.pdf',
                     color='Red', linestyle='-.', linewidth=1.0)

# Additionally, JEOL_pts objects can be saved as HDF5 files. This has the benefit
# that all attributes (drift_images, parameters) are also stored. Use base name of
# original file and pass along keywords to `h5py.create_dataset()`.
>>>> dc.save_hdf5(compression='gzip', compression_opts=9)

# Initialize from HDF5 file. Only filename is used, additional keywords
# are ignored.
>>>> dc3 = JEOL_pts('data/128.h5')
>>>> dc3.parameters
{'PTTD Cond': {'Meas Cond': {'CONDPAGE0_C.R': {'Tpl': {'index': 3,
     'List': ['T1', 'T2', 'T3', 'T4']},
        .
        .
        .
    'FocusMP': 16043213}}}}

# Read spectral data from '.eds' file.
>>>> s = JEOL_spectrum('data/spot.eds')

>>>> s.file_name
'data/spot.eds'

>>>> s.file_date
'2022-02-17 15:15:20'

# Display meta data.
# First header
>>>> s.header
{'sp_name': '001',
 'username': 'JEM Administrator',
 'arr': array([0.e+00, 1.e+01, 1.e+00, 0.e+00, 1.e+03, 1.e+02, 1.e+00, 1.e+05,
        0.e+00, 0.e+00]),
 'Esc': 1.75,
 'Fnano F': 0.12,
 'E Noise': 45.0,
 'CH Res': 0.01,
 'live time': 30.0,
 'real time': 30.84,
 'DeadTime': 2.0,
 'CountRate': 1238.0,
 'CountRate n': 58,
 'CountRate sum': array([6.8256000e+04, 8.4252794e+07]),
 'CountRate value': 1176.8275862068965,
 'DeadTime n': 58,
 'DeadTime sum': array([150., 412.]),
 'DeadTime value': 2.586206896551724,
 'CoefA': 0.0100006,
 'CoefB': -0.00122558,
 'State': 'Live Time',
 'Tpl': 'T4',
 'NumCH': 4096}

# Now footer.
>>>> s.footer
{'Excluded elements': array([  1,   2,   3,   4,  10,  18,  36,  43,  54,  61,  84,  85,  86,
         87,  88,  89,  91,  93,  94,  95,  96,  97,  98,  99, 100, 101,
        102, 103], dtype=uint16),
 'Selected elements': {'O K': {'Z': 8,
   'Roi_min': 46,
        .
        .
        .
   'SpatZ': 79,
   'SpatThic': 0.015000000596046448,
   'SiDead': 0.09999999403953552,
   'SiThic': 0.5}}

# Size of spectral data.
>>>> s.data.shape
(4096,)

# Plot (uncalibrated) data
>>>> JU.plot_spectrum(s.data,
                      E_range=(0, 20),
                      M_ticks=(4, 1))

# If you need the calibrated data (x-axis)
>>>> x = range(s.data.shape[0]) * s.header['CoefA'] + s.header['CoefB']

# PointLine data (a series on points located on an arbitrarily defined line)
# can be loaded via the accompanying '.pln' file.
>>>> pl = JEOL_PointLine('data/PointLine/View000_0000001.pln')

# Image file used in the definition of the line.
>>>> pl.Image_name
'View000_0000000.img'

# Full image object (`JEOL_eds.Image`) is stored with all its attributes.
>>>> pl.ref_image
<JEOL_eds.JEOL_eds.JEOL_image at 0x7fd6963d53d0>

>>>> pl.ref_image.nm_per_pixel
1.93359375

# Plot image with scale bar.
>>>> scale_bar = {'label': '200nm',
                  'position': 'upper left',
                  'f_calib': pl.ref_image.nm_per_pixel,
                  'color': 'black'}
>>>> JU.plot_map(pl.ref_image.image, 'inferno_r', scale_bar=scale_bar)

# `JEOL_PointLine.eds_dict` is a dict with marker as key and a list
# [FileName, xPos, yPos] as content.
>>>> pl.eds_dict
{0: ['View000_0000006.eds', 85.3125, 96.4375],
 1: ['View000_0000005.eds', 81.4375, 92.6875],
 2: ['View000_0000004.eds', 77.5625, 88.9375],
 3: ['View000_0000003.eds', 73.6875, 85.1875],
 4: ['View000_0000002.eds', 69.8125, 81.4375]}

# Meta data of the first file in `JEOL_PointList.eds_list`. Most important
# values should be ['CoefA'], ['CoefB'] (calibration of energy axis).
>>>> pl.eds_header
{'sp_name': 'LG10004 ; 41.676 nm',
        .
        .
        .
 'CoefA': 0.0100006,
 'CoefB': -0.00122558,
 'State': 'Live Time',
 'Tpl': 'T4',
 'NumCH': 4096}

# Check definition of PointLine (points and markers).
>>>> pl.show_PointLine(ROI=(45,110,50,100),
                                   color='red', ann_color='blue')

# Plot spectrum with marker '2', i.e. the third in the list.
>>>> JU.plot_spectrum(pl.eds_data[2])

# Extract profile of total x-ray intensity.
>>>> p_tot = pl.profile()

# extract profile of Ti Ka line with one spectrum (marker '2') omitted,
# i.e. replaced by `NaN`.
>>>> p_Ti = pl.profile(interval=(4.4, 4.65),
                       energy=True,
                       markers=[0, 1, 3, 4])
```

## Bugs

Parameters loaded from '.pts' files might have different types than the ones
loaded from '.h5' files. Thus take extra care if you need to compare them:

```python
# Load and store as hdf5.
>>>> dc = JEOL_pts('data/128.pts')
>>>> dc.save_hdf5(compression='gzip', compression_opts=9)
# Initialize from hdf5
>>>> dc_hdf5 = JEOL_pts('data/128.h5')

# Compare parameters dict gives unexpected result.
>>>> p = dc.parameters['PTTD Data']['AnalyzableMap MeasData']['MeasCond']
>>>> p_hdf5 = dc_hdf5.parameters['PTTD Data']['AnalyzableMap MeasData']['MeasCond']
>>>> p == p_hdf5
False

# But they seem identical.
>>>> p
{'AccKV': 200.0,
 'AccNA': 7.475,
 'Mag': 800000,
 'WD': 3.2,
 'ScanR': 270.0,
 'FocusMP': 16043213}

>>>> p_hdf5
{'AccKV': 200.0,
 'AccNA': 7.475,
 'Mag': 800000,
 'WD': 3.2,
 'ScanR': 270.0,
 'FocusMP': 16043213}

# The issue is different types.
# This works.
>>>> p['AccKV'] == p_hdf5['AccKV']
True
>>>> type(p['AccKV'])
numpy.float32
>>>> type(p_hdf5['AccKV'])
float

# This causes the issue.
>>>> p['AccNA'] == p_hdf5['AccNA']
False
>>>> type(p['AccNA'])
numpy.float32
>>>> type(p_hdf5['AccNA'])
float
```

The position in the '.img' file where **spot**, **area** ('.eds.') data were acquired is still unknown (they must be present in the data sets but I have not yet been able to extract them).
