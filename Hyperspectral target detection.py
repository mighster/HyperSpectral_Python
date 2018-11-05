# imports and setup
import numpy as np
import os.path
import scipy.io
from loadmat import loadmat

import matplotlib as mpl
%matplotlib inline
default_dpi = mpl.rcParamsDefault['figure.dpi']
mpl.rcParams['figure.dpi'] = default_dpi*2
import matplotlib.pyplot as plt

from hsi_detectors import smf_detector,ace_detector

# load gulfport campus image
img_fname = 'muufl_gulfport_campus_w_lidar_1.mat'
spectra_fname = 'tgt_img_spectra.mat'

dataset = loadmat(img_fname)['hsi']

hsi = dataset['Data']
n_r,n_c,n_b = hsi.shape
wvl = dataset['info']['wavelength']
rgb = dataset['RGB']

# load the target signatures
spectra_dataset = loadmat(spectra_fname)
tgts = spectra_dataset['tgt_img_spectra']['spectra']
tgt_names = spectra_dataset['tgt_img_spectra']['names']

# check out the shape of the targets array
tgts.shape
# check out the target values
tgts

# look at the target names
tgt_names

# plot the target signature spectra
plt.plot(wvl,tgts)
plt.legend(tgt_names)

# select the spectra for the Brown target, try to find it using ACE
br_sig = tgts[:,0]

conf = ace_detector(hsi,br_sig)

# plot the ACE confidence map
plt.imshow(conf)
plt.colorbar()
plt.title('ACE Confidence Map')

# threshold to confidence map to make target declrations
plt.imshow(conf > 0.5)

# run the same signature in Spectral Matched Filter
conf_smf = smf_detector(hsi,br_sig)
plt.imshow(conf_smf)
plt.colorbar()
plt.title('SMF Confidence Map')

# threshold to make declrations
plt.imshow(conf_smf > 5)

# load the ground truth target locations
gt = dataset['groundTruth']
gt_row,gt_col,gt_name = gt['Targets_rowIndices'],gt['Targets_colIndices'],gt['Targets_Type']

br_rc = [(row,col) for row,col,name in zip(gt_row,gt_col,gt_name) if name == 'brown']

# inspect
br_rc

# plot the ground truth over the declaration map
plt.subplot(1,2,1)
plt.imshow(conf_smf > 5)
for r,c in br_rc:
    plt.plot(c,r,'rx')
plt.subplot(1,2,2)
plt.imshow(conf_smf > 5)

