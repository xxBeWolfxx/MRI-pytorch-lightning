import cv2

# read image
from rich import print

img = cv2.imread('D:/Projekty/Git projekt/mastery-machine-learning/skullstripper_data/z_train/img/subdir_required_by_keras/CC0001_philips_15_55_Mimage-slice003.png', cv2.IMREAD_UNCHANGED)
mask = cv2.imread('D:/Projekty/Git projekt/mastery-machine-learning/skullstripper_data/z_train/mask/subdir_required_by_keras/CC0001_philips_15_55_Mimage-slice003.png', cv2.IMREAD_UNCHANGED)

# get dimensions of image
dimensions = img.shape
dimensions_mask = mask.shape
# height, width, number of channels in image
height = img.shape[0]
width = img.shape[1]
channels = len(img.shape)
if channels == 2:

    channels = 1

if channels == 3:

    channels = img.shape[-1]

print('Image Dimension    : ', dimensions, dimensions_mask)
print('Image Height       : ', height)
print('Image Width        : ', width)
print('Number of Channels : ', channels)

