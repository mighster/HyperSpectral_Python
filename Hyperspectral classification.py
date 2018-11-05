# imports and setup
import numpy as np
import os.path
import scipy.io
from loadmat import loadmat

import matplotlib as mpl
default_dpi = mpl.rcParamsDefault['figure.dpi']
mpl.rcParams['figure.dpi'] = default_dpi*1.5
import matplotlib.pyplot as plt

# load gulfport campus image (with labels)
img_fname = 'muufl_gulfport_campus_1_hsi_220_label.mat'

dataset = loadmat(img_fname)['hsi']

hsi = dataset['Data']
n_r,n_c,n_b = hsi.shape
wvl = dataset['info']['wavelength']
rgb = dataset['RGB']

# plot the RGB image to see what we are looking at
plt.imshow(rgb)

# pull label info from the dataset
gt = dataset['sceneLabels']
label_names = gt['Materials_Type']
label_img = gt['labels']

# inspect the label values
print('min label value:',label_img.min())
print('max label value:',label_img.max())
print('label names:',label_names)

# show the labels as an image
def discrete_matshow(data,minv=None,maxv=None,lbls=None):
    #https://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar
    #get discrete colormap
    if minv is None:
        minv = np.min(data)
    if maxv is None:
        maxv = np.max(data)
    cmap = plt.get_cmap('RdBu', maxv-minv+1)
    # set limits .5 outside true range
    newdata = data.copy().astype(float)
    newdata[data > maxv] = np.nan
    newdata[data < minv] = np.nan
    mat = plt.matshow(newdata,cmap=cmap,vmin = minv-.5, vmax = maxv+.5)
    #tell the colorbar to tick at integers
    cax = plt.colorbar(mat, ticks=np.arange(minv,maxv+1))
    
discrete_matshow(label_img,1,10)

# train a nearest neighbor classifier with N samples from each of 10 classes (1-10)
from sklearn.neighbors import KNeighborsClassifier

# construct the training set
samples = []
labels = []
n_samp_per = 100
for i in range(1,11):
    lbl_inds = (label_img == i).nonzero()
    n_inds = lbl_inds[0].shape[0]
    ns = min(n_inds,n_samp_per)
    perm = np.random.permutation(np.arange(n_inds))
    perm_lbl = (lbl_inds[0][perm],lbl_inds[1][perm])
    pix = hsi[perm_lbl[0][:ns],perm_lbl[1][:ns],:]
    lbls = np.full((ns,1),i,dtype=int)
    samples.append(pix)
    labels.append(lbls)
    
X = np.vstack(samples)
y = np.vstack(labels).squeeze()

# fit the KNN classifer
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)

# predict outputs
M = np.reshape(hsi,(n_r*n_c,n_b))
Z = knn.predict(M)

# reshape back into an image and dispay results
z_img = np.reshape(Z,(n_r,n_c))
discrete_matshow(z_img)

# show the training data again for comparison
print(label_names)
discrete_matshow(label_img,1,10)

# evaluate performance on training set using confusion matrix, accuracy score
from sklearn.metrics import confusion_matrix

lbl_array = np.reshape(label_img,(n_r*n_c))
valid_inds = np.logical_and(lbl_array > 0,lbl_array < 11)

cm = confusion_matrix(lbl_array[valid_inds],Z[valid_inds])
row_sum = cm.sum(axis=1).reshape((cm.shape[0],1)) #sum of rows as column vector
norm_cm = cm/row_sum

# compute overall accuracy, 
oa = np.diag(cm).sum()/cm.sum()
print('overall accuracy: %.3f'%oa)
plt.imshow(norm_cm)
plt.colorbar()
plt.title('Per-class confusion')
