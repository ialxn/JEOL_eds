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
(pip for installation)
```

Download zip and extract or clone repository. From the resulting folder run

```bash
$ pip install .
```

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
# as tuple (left, right, top, bottom) of pixels for selected frames.
>>>> plt.plot(dc.spectrum(ROI=(10, 20, 50, 100), frames=[0,1,2,10,11,12,30,31,32]))
<matplotlib.lines.Line2D at 0x7f7192b58050>


# Make movie of drift_images and total EDS intensity and store it
# as 'test/128.mp4'.
>>>> dc = JEOL_pts('test/128.pts', split_frames=True, read_drift=True)
>>>> dc.make_movie()
```

