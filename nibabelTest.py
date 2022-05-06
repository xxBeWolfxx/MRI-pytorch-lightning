import matplotlib.image

import nibabel as nib
from PIL import Image
import scipy.misc
import os
import numpy as np
from matplotlib import cm
from nibabel.testing import data_path

# img = nib.load("D:/Projekty/Git projekt/mastery-machine-learning/nii/WMH Segmentation Challenge/Amsterdam_GE3T/GE3T/100/pre/3DT1.nii.gz")

path = 'D:/Projekty/Git projekt/mastery-machine-learning/nii/WMH Segmentation Challenge/Amsterdam_GE3T/GE3T/'
pathToSaveImg = 'D:/Projekty/Git projekt/mastery-machine-learning/nii_data/I/img'

listOfPatient = os.listdir(path)

# img_fdata = img.get_fdata()
# y,z,x = img.shape

for pp in listOfPatient:
    newPathImg = path + f"{pp}/pre/FLAIR.nii.gz"
    img = nib.load(newPathImg)
    img_fdata = img.get_fdata()
    y, z, x = img.shape
    for i in range(z):
        slice = img_fdata[:, i, :]
        matplotlib.image.imsave(f'D:/Projekty/Git projekt/mastery-machine-learning/nii_data/I/img/{pp}imgZ{i}.png', slice, cmap=cm.gray)
    # for j in range(y):
    #     slice = img_fdata[j, :, :]
    #     matplotlib.image.imsave(f'D:/Projekty/Git projekt/mastery-machine-learning/nii_data/I/img/{pp}imgY{j}.png',
    #                             slice, cmap=cm.gray)
    # for k in range(x):
    #     slice = img_fdata[:, :, k]
    #     matplotlib.image.imsave(f'D:/Projekty/Git projekt/mastery-machine-learning/nii_data/I/img/{pp}imgX{k}.png',
    #                             slice, cmap=cm.gray)

    newPathMask = path + f"{pp}/wmh.nii.gz"
    img = nib.load(newPathMask)
    img_fdata = img.get_fdata()
    y, z, x = img.shape
    for i in range(z):
        slice = img_fdata[:, i, :]
        matplotlib.image.imsave(f'D:/Projekty/Git projekt/mastery-machine-learning/nii_data/I/mask/{pp}imgZ{i}.png',
                                slice, cmap=cm.gray)
    # for j in range(y):
    #     slice = img_fdata[j, :, :]
    #     matplotlib.image.imsave(f'D:/Projekty/Git projekt/mastery-machine-learning/nii_data/Ia/mask/{pp}imgY{j}.png',
    #                             slice, cmap=cm.gray)
    # for k in range(x):
    #     slice = img_fdata[:, :, k]
    #     matplotlib.image.imsave(f'D:/Projekty/Git projekt/mastery-machine-learning/nii_data/Ia/mask/{pp}imgX{k}.png',
    #                             slice, cmap=cm.gray)

# matplotlib.image.imsave('D:/Projekty/Git projekt/mastery-machine-learning/mask11.png', slice, cmap=cm.gray)




