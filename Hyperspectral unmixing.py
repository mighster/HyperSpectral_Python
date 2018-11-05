# imports and setup
import numpy as np
import os.path
import scipy.io
from loadmat import loadmat

import matplotlib as mpl
default_dpi = mpl.rcParamsDefault['figure.dpi']
mpl.rcParams['figure.dpi'] = default_dpi*2
import matplotlib.pyplot as plt

# load gulfport campus image
img_fname = 'muufl_gulfport_campus_w_lidar_1.mat'
spectra_fname = 'tgt_img_spectra.mat'

dataset = loadmat(img_fname)['hsi']

hsi = dataset['Data'][:,:,4:-4] # trim noisy bands 
valid_mask = dataset['valid_mask'].astype(bool)
n_r,n_c,n_b = hsi.shape
wvl = dataset['info']['wavelength'][4:-4]
rgb = dataset['RGB']

# extract some endmembers using Pixel Purity Index algorithm
#  using PySptools from https://pysptools.sourceforge.io
import pysptools
import pysptools.eea

hsi_array = np.reshape(hsi,(n_r*n_c,n_b))
valid_array = np.reshape(valid_mask,(n_r*n_c,))
M = hsi_array[valid_array,:]
q = 3
numSkewers = 500
E,inds = pysptools.eea.eea.PPI(M, q, numSkewers)

# plot the endmembers we found
plt.plot(wvl,E.T)
plt.xlabel('wavelength (nm)')
plt.ylabel('reflectance')
plt.legend([str(i+1) for i in range(q)])
plt.title("PPI Endmembers")

# find abundances given the endmembers
import pysptools.abundance_maps

maps = pysptools.abundance_maps.amaps.FCLS(M, E)
#maps = np.zeros((M.shape[0],E.shape[1]))

# re-ravel abundance maps
map_imgs = []
for i in range(q):
    map_lin = np.zeros((n_r*n_c,))
    map_lin[valid_array] = maps[:,i]
    map_imgs.append(np.reshape(map_lin,(n_r,n_c)))
	
# display abundance maps
for i in range(q):
    plt.figure()
    plt.imshow(map_imgs[i],vmin=0,vmax=1)
    plt.colorbar()
    plt.title('FCLS Abundance Map %d'%(i+1,))
	
	# run SPICE to find number of endmembers, endmembers, and abundances simultaneously
from SPICE import *

params = SPICEParameters()
inputData = M.T.astype(float)
# to save time, downsample inputData
dsData = inputData[:,::20]
dsData.shape

# run SPICE
[eM,dsP] = SPICE(dsData,params)

# unmix endmembers again with full data matrix (because we downsampled for sake of time)
P = unmix2(inputData,eM)
n_em = eM.shape[1]

#plot endmembers
plt.plot(wvl,eM)
plt.xlabel('wavelength (nm)')
plt.ylabel('reflectance')
plt.legend([str(i+1) for i in range(5)])
plt.title('SPICE Endmembers')

# re-reval abundance maps
P_imgs = []
for i in range(n_em):
    map_lin = np.zeros((n_r*n_c,))
    map_lin[valid_array] = P[:,i]
    P_imgs.append(np.reshape(map_lin,(n_r,n_c)))
	
	# display abundance maps
for i in range(n_em):
    plt.figure()
    plt.imshow(P_imgs[i],vmin=0,vmax=1)
    plt.colorbar()
    plt.title('SPICE Abundance Map %d'%(i+1,))
	
