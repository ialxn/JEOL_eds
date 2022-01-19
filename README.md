# JEOL_eds

A python module to read binary data files ('.pts') or image file ('.img') by JEOL's Analysis Station software. The function to parse the header of the binary file was copied from HyperSpy (hyperspy/io_plugins/jeol.py scheduled for inclusion into HyperSpy 1.7).

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
>>>> from JEOL_eds import JEOL_pts

# Read EDS data
>>>> dc = JEOL_pts('128.pts', split_frames=True, E_cutoff=11.0)

# If 'split_frames=True' is used and the data cube becomes
# too big to be kept in memory a subset of frames can be read
# by using the keyword parameter 'list_frames'.
>>>> small_dc = JEOL_pts('128.pts',
                         split_frames=True, list_frames=[1,2,4,8,16],
                         E_cutoff=11.0)
>>>> small_dc.frame_list
[1, 2, 4, 8, 16]

>>>> small_dc.dcube.shape
(5, 128, 128, 1100)
# The frames in the data cube correspond to the original frames 1, 2, 4, 8, and 16.


# Read and plot image data
>>>> from JEOL_eds import JEOL_image
>>>> import matplotlib.pyplot as plt
>>>> demo = JEOL_image('data/demo.img')
>>>> plt.imshow(demo.image)
<matplotlib.image.AxesImage at 0x7fa08425d350>

# Use ``plot_map()`` for more features.
# 'data/demo.img' is a BF image. Thus use inverted color map.
>>>> scale_bar = {'label': '200nm',
                  'f_calib': demo.nm_per_pixel,
                  'color': 'white'}
>>>> plot_map(demo.image, 'inferno_r', scale_bar=scale_bar)


# Report meta data of image file
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
# This is only available for '*.map' files.
>>>> demo.pixel_size
0.99


# To read (and process) large data sets you might use the following code fragment.
# Get number of frames (read only meta data to speed up this step).
>>>> large_fn = "data/128.pts"
>>>> large = JEOL_pts(large_fn, only_metadata=True)
>>>> N = large.parameters['EDS Data']['AnalyzableMap MeasData']['Doc']['Sweep']
>>>> per_batch = 10
>>>> N_batches = N // per_batch
>>>> for i in range(N_batches):
         flist = [i*per_batch + j for j in range(per_batch)]
         subset = JEOL_pts(large_fn, split_frames=True, frame_list=flist)
         # Do the processing of the subset. Here print total X-ray intensity.
         print(subset.map().sum())


# Cu Kalpha map of all even frames.
>>>> m = dc.map(interval=(7.9, 8.1),
                energy=True,
                frames=range(0, dc.dcube.shape[0], 2))

# Cu Kalpha map of frames 0..10. Frames are aligned using
# frame 5 as reference. Wiener filtered frames are used to
# calculate the shifts.
# Verbose output
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


# Plot spectrum integrated over full image.
# If option 'split_frames' was used to read the data the
# following plots the sum spectrum of all frames added.
>>>> plt.plot(dc.spectrum())
[<matplotlib.lines.Line2D at 0x7f7192feec10>]

# The sum spectrum of the whole data cube is also stored
# in the raw data and can be accessed much quicker.
>>>> plt.plot(dc.ref_spectrum)
[<matplotlib.lines.Line2D at 0x7f3131a489d0>]

# Plot sum spectrum corresponding to a (rectangular) ROI specified
# as tuple (top, bottom, left, light) of pixels for selected frames.
# Verify definition of ROI before you apply it using the total x-ray
# intensity as image.
>>>> ROI = (10, 20, 50, 100)
>>>> show_ROI(dc.map(), ROI, alpha=0.6)
>>>> plt.plot(dc.spectrum(ROI=ROI, frames=[0,1,2,10,11,12,30,31,32]))
<matplotlib.lines.Line2D at 0x7f7192b58050>


# Create overlay of elemental maps
>>>> from JEOL_eds.utils import create_overlay

# Load data. Data does not contain drift images and all frames were
# added, thus only a single frame is present.
>>>> dc = JEOL_pts('data/complex_oxide.h5')

# Extract some elemental maps. Where possible, dd contribution of
# several lines.
>>>> Ti = dc.map(interval=(4.4, 5.1), energy=True)      # Ka,b
>>>> Fe = dc.map(interval=(6.25, 6.6), energy=True)     # Ka
>>>> Sr = dc.map(interval=(13.9, 14.4), energy=True)    # Ka
>>>> Co = dc.map(interval=(6.75, 7.0), energy=True)     # Ka
>>>> Co += dc.map(interval=(7.5, 7.8), energy=True)     # Kb
>>>> O = dc.map(interval=(0.45, 0.6), energy=True)

# Visualize the CoFeOx distribution using first of the `drift_images`
# as background. Note that drift images were not stored in the data
# supplied and this will raise a TypeError.
>>>> create_overlay([Fe, Co],
                    ['Maroon', 'Violet'],
                    legends=['Fe', 'Co'],
                    BG_image=dc.drift_images[0])


# Plot spectra
>>>> from JEOL_eds.utils import plot_spectrum

# Plot and save reference spectrum between 1.0 and 2.5 keV.
# Plot one minor tick on x-axis and four on y-axis. Pass
# some keywords to `matplotlib.pyplot.plot()`.
>>>> plot_spectrum(dc.ref_spectrum,
                   E_range=(4, 17.5),
                   M_ticks=(1, 4),
                   outfile='ref_spectrum.pdf',
                   color='Red', linestyle='-.', linewidth=1.0)


# To insert the output of the different plot functions imported from
# `JEOL_eds.utils` into a sub-plot, use the following code fragment
fig, (ax1, ax2) = plt.subplots(1, 2)
# Use `ax1` for overlay
>>>> plt.sca(ax1)
>>>> create_overlay((O, Sr, Ti),
                    ('Blue', 'Green', 'Red'),
                    legends=['O', 'Sr', 'Ti'],
                    BG_image=dc.drift_images[0])
# Use `ax2` for spectrum
>>>> plt.sca(ax2)
>>>> plot_spectrum(dc.ref_spectrum, E_range=(4, 17.5))
>>>> plt.tight_layout() 	# Prevents overlapping labels
>>>> plt.savefig('demo.pdf')


# Calculate and plot line profiles.
>>>> from JEOL_eds.utils import show_line, get_profile

# Extract carbon map
>>>> C_map = dc.map(interval=(0.22, 0.34), energy=True)

# Define line. Verify definition.
>>>> line = (80, 5, 110, 100)
>>>> width = 10
>>>> show_line(C_map, line, linewidth=width, cmap='inferno')

# Calculate profile along a given line (width equals 10 pixels) and
# plot it.
>>>> profile = get_profile(C_map, line, linewidth=width)
>>>> plt.plot(profile)


# Make movie of drift_images and total EDS intensity and store it
# as 'data/128.mp4'.
>>>> dc = JEOL_pts('data/128.pts', split_frames=True, read_drift=True)
>>>> dc.make_movie()


# Check for contamination by carbon.
# Integrate carbon Ka line.
>>>> ts = dc.time_series(interval=(0.45, 0.6), energy=True)
# Plot and save the time series.
>>>> from JEOL_eds.utils import plot_tseries
>>>> plot_tseries(ts,
                  M_ticks=(9,4),
                  outfile='carbon_Ka.pdf',
                  color='Red', linestyle='-.', linewidth=1.0)


# Additionally, JEOL_pts object can be saved as hdf5 files.
# This has the benefit that all attributes (drift_images, parameters)
# are also stored.
# Use base name of original file and pass along keywords to
# `h5py.create_dataset()`.
>>>> dc.save_hdf5(compression='gzip', compression_opts=9)

# Initialize from hdf5 file. Only filename is used, additional keywords
# are ignored.
>>>> dc3 = JEOL_pts('128.h5')
>>>> dc3.parameters
{'PTTD Cond': {'Meas Cond': {'CONDPAGE0_C.R': {'Tpl': {'index': 3,
     'List': ['T1', 'T2', 'T3', 'T4']},
.
.
.
    'FocusMP': 16043213}}}}
```

## Bugs

Parameters loaded from '.pts' might have different types than the ones
loaded from 'h5' files. Thus take extra care if you need to compare them:
```python
# Load and store as hdf5.
>>>> dc = JEOL_pts('128.pts')
>>>> dc.save_hdf5(compression='gzip', compression_opts=9)
# Initialize from hdf5
>>>> dc_hdf5 = JEOL_pts('128.h5')

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
````
