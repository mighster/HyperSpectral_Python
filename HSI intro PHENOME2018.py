# imports and setup
import numpy as np
import os.path
import scipy.io
from loadmat import loadmat

import matplotlib.pyplot as plt
import matplotlib as mpl
%matplotlib inline
default_dpi = mpl.rcParamsDefault['figure.dpi']
mpl.rcParams['figure.dpi'] = default_dpi*2

# load gulfport campus image
img_fname = 'muufl_gulfport_campus_w_lidar_1.mat'
spectra_fname = 'tgt_img_spectra.mat'

dataset = loadmat(img_fname)['hsi']

hsi = dataset['Data']

# check out the shape of the data
n_r,n_c,n_b = hsi.shape
hsi.shape

# pull a 'random' pixel/spectrum
rr,cc = 150,150
spectrum = hsi[rr,cc,:]
spectrum

# plot a spectrum
plt.plot(spectrum)

# That last plot would make your advisor sad.
#  Label your AXES!
wavelengths = dataset['info']['wavelength']

plt.plot(wavelengths,spectrum)
plt.xlabel('wavelength (nm)')
plt.ylabel('% reflectance')
plt.title('A Spectrum')

# plot an image of an individual band
plt.imshow(hsi[:,:,30],vmin=0,vmax=.75,cmap='Reds')
plt.colorbar()
plt.title('A single band of Hyperspectral Image in False Color')

# find the band numbers for approximate R,G,B wavelengths
#blue           #green          #red
wavelengths[9],wavelengths[20],wavelengths[30]

# make a psuedo-RGB image from appropriate bands
psuedo_rgb = hsi[:,:,(30,20,9)]
psuedo_rgb = np.clip(psuedo_rgb,0,1.0) 
plt.imshow(psuedo_rgb)

# Thats too dark. Add some gamma correction
plt.imshow(psuedo_rgb**(1/2.2)) 

# compare to the provided RGB image (made with better band selection/weighting)
plt.imshow(dataset['RGB'])
plt.plot(rr,cc,'m*') #label our selected location

