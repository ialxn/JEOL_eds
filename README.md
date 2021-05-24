# JEOL_eds

A python module to read binary data files ('.pts') by JEOL's Analysis Station software. The function to parse the header of the binary file was copied from HyperSpy (hyperspy/io_plugins/jeol.py scheduled for inclusion into HyperSpy 1.7).

This module does not aim to replace HyperSpy which is much more feature-rich. Instead it provides an easy interface to extract spectra or elemental maps from the binary file much like the *Play Back* feature in **Analysis Station**.

## Installation

### Requirements
```
Python 3.6+
numpy
scipy
matplotlib
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

# read EDS data
>>>> dc = JEOL_pts('128.pts', split_frames=True, E_cutoff=11.0)


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
>>>> dc = JEOL_pts('test/complex_oxide.h5')

# Extract some elemental maps. Where possible, dd contribution of
# several lines.
>>>> Ti = dc.map(interval=(4.4, 5.1), energy=True)      # Ka,b
>>>> Fe = dc.map(interval=(6.25, 6.6), energy=True)     # Ka
>>>> Sr = dc.map(interval=(13.9, 14.4), energy=True)    # Ka
>>>> Co = dc.map(interval=(6.75, 7.0), energy=True)     # Ka
>>>> Co += dc.map(interval=(7.5, 7.8), energy=True)     # Kb
>>>> O = dc.map(interval=(0.45, 0.6), energy=True)

# Visualize the CoFeOx distribution using first of the `drift_images`
# as background. Note that drift images were not stored in the test
# data supplied and this will raise a TypeError.
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


# Make movie of drift_images and total EDS intensity and store it
# as 'test/128.mp4'.
>>>> dc = JEOL_pts('test/128.pts', split_frames=True, read_drift=True)
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
