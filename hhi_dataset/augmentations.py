##
# ingroup: ML_Utilities
# file:    augmentations.py
# brief:   This file contains image augmentation functions
# author:  Nikita Kovalenko (mykyta.kovalenko@hhi.fraunhofer.de)
# date:    03.09.2020
#
# Copyright:
# 2020 Fraunhofer Institute for Telecommunications, Heinrich-Hertz-Institut (HHI)
# The copyright of this software source code is the property of HHI.
# This software may be used and/or copied only with the written permission
# of HHI and in accordance with the terms and conditions stipulated
# in the agreement/contract under which the software has been supplied.
# The software distributed under this license is distributed on an "AS IS" basis,
# WITHOUT WARRANTY OF ANY KIND, either expressed or implied.
##
#----------------------
import numpy as np
import cv2
import copy
import math
#----------------------
from hhi_dataset.tools import *
#----------------------
# HORIZONTAL FLIP
def augment_flip(image, max_chance=0.5, horizontal=True, vertical=False):
    """
    Performs horizontal and vertical flips of the image

    Args:
        image (numpy.ndarray): Input image in the Numpy format
        max_chance (float, optional): Occurence probability threshold. Defaults to 0.5.
        horizontal (bool, optional): Perform horizontal flip?. Defaults to True.
        vertical (bool, optional): Perform vertical flip?. Defaults to False.
    """
    chance = np.random.rand()
    if chance > max_chance:
        return

    if horizontal:
        image[:] = cv2.flip(image, 1)
    if vertical:
        image[:] = cv2.flip(image, 0)

# HORIZONTAL AND VERTICAL TRANSLATION
def augment_translate(image, max_chance=0.5, max_range=0.1, borderValue=128):
    """Performs an affine translation of the image

    Args:
        image (numpy.ndarray): Input image in the Numpy format
        max_chance (float, optional): Occurence probability threshold. Defaults to 0.5.
        max_range (float, optional): Maximum range, as a fraction of the image size, 
        e.g. value '0.2' means a translation of up to 20% of the image size. 
        Can be given as one value (0.2) or as a range (-0.2, 0.2). Defaults to 0.1.
        borderValue (int, optional): Border value for padding. Defaults to 128.
    """
    chance_x = np.random.rand()
    chance_y = np.random.rand()

    if not isinstance(max_range, list):
        max_range = [-max_range, max_range]

    if image.ndim > 2 and not isinstance(borderValue, list):
        borderValue = tuple([borderValue] * image.shape[-1])

    h,w,ch = image.shape[:2] + (image.shape[-1] if image.ndim > 2 else 0,)

    x_tr = np.random.uniform(max_range[0], max_range[1]) if chance_x < max_chance else 0
    y_tr = np.random.uniform(max_range[0], max_range[1]) if chance_y < max_chance else 0

    image[:] = cv2.warpAffine(image, np.float32([[1,0,w*x_tr],[0,1,h*y_tr]]), (w,h),
                           borderMode=cv2.BORDER_CONSTANT, borderValue=borderValue)

# UP- AND DOWNSCALING (with unchanged image size)
def augment_scale(image, max_chance=0.5, max_range=0.1, borderValue=128):
    h,w,ch = image.shape[:2] + (image.shape[-1] if image.ndim > 2 else 0,)

    if not isinstance(max_range, list):
        max_range = [-max_range, max_range]

    if image.ndim > 2 and not isinstance(borderValue, list):
        borderValue = tuple([borderValue] * image.shape[-1])

    fx = np.random.uniform(1+max_range[0], 1+max_range[1]) if np.random.rand() < max_chance else 1
    fy = np.random.uniform(1+max_range[0], 1+max_range[1]) if np.random.rand() < max_chance else 1
    f = [fx,fy]

    tr_x = (w - (w * fx)) // 2
    tr_y = (h - (h * fy)) // 2

    image[:] = cv2.warpAffine(image, np.float32([[fx,0,tr_x],[0,fy,tr_y]]), (w,h),
                        borderMode=cv2.BORDER_CONSTANT, borderValue=borderValue)

# SHEARING
def augment_shear(image, max_chance=0.5, max_range=0.2, horizontal=True, vertical=False, borderValue=128):
    chance_x = np.random.rand()
    chance_y = np.random.rand()

    if not isinstance(max_range, list):
        max_range = [-max_range, max_range]

    if image.ndim > 2 and not isinstance(borderValue, list):
        borderValue = tuple([borderValue] * image.shape[-1])

    h,w,ch = image.shape[:2] + (image.shape[-1] if image.ndim > 2 else 0,)

    x_tr = np.random.uniform(max_range[0], max_range[1]) if (chance_x < max_chance and horizontal) else 0
    y_tr = np.random.uniform(max_range[0], max_range[1]) if (chance_y < max_chance and vertical) else 0

    M2 = np.float32([[1, x_tr, 0], [y_tr, 1, 0]])
    M2[0,2] = -M2[0,1] * w/2
    M2[1,2] = -M2[1,0] * h/2
    image[:] = cv2.warpAffine(image, np.float32(M2), (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=borderValue)

# ROTATION
def augment_rotate(image, max_chance=0.5, max_angle=20, borderValue=128):
    h,w,ch = image.shape[:2] + (image.shape[-1] if image.ndim > 2 else 0,)
    chance = np.random.rand()

    if image.ndim > 2 and not isinstance(borderValue, list):
        borderValue = tuple([borderValue] * image.shape[-1])

    if not isinstance(max_angle, list):
        max_angle = [-max_angle, max_angle]

    angle = np.random.uniform(max_angle[0], max_angle[1]) if chance < max_chance else 0

    image[:] = __rotateImage(image, angle, borderValue)

# helper: image rotation
def __rotateImage(image, angle, borderValue):
    image_center = tuple(np.array(image.shape[:2][::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[:2][::-1], flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=borderValue)
    return result

# ADD ARTIFACTS LIKE IF THE IMAGE WAS RESIZED
def augment_add_resize_artifacts(image, max_chance=0.5, max_factor = 8):
    f = np.random.uniform(1, max_factor + 0.01) if np.random.rand() < max_chance else 1
    if np.random.rand() < max_chance: f = 1 / f
    img_r = cv2.resize(image, None, fx=1/f, fy=1/f)
    image[:] = cv2.resize(img_r, (image.shape[1], image.shape[0]))

# ADD RANDOM GAUSSIAN NOISE
def augment_add_noise(image, max_chance=0.5, max_intensity=0.5):
    factor = 100
    intensity = np.random.uniform(0.001, max_intensity) if np.random.rand() < max_chance else 0
    img_n = np.random.uniform(-factor*intensity, factor*intensity, image.size).reshape(image.shape)

    image[:] = np.uint8(np.clip(np.float32(image) + img_n, 0, 255))

# CHANGE CONTRAST
def augment_contrast(image, max_chance=0.5, max_range=0.2):
    if np.random.rand() < max_chance:
        rng = np.random.uniform(1-max_range, 1+max_range)
        rng = np.clip(rng, 0.3, 3.0)
        image[:] = np.clip( image * rng, 0, 255 )

# CHANGE BRIGHTNESS
def augment_brightness(image, max_chance=0.5, max_range=100):
    if np.random.rand() < max_chance:
        rng = np.random.uniform(-max_range, max_range)
        image[:] = np.clip( image + rng, 0, 255 )

# CHANGE GAMMA
def augment_gamma(image, max_chance=0.5, max_range=(0.05, 10)):
    if np.random.rand() < max_chance:
        rng = np.random.uniform(max_range[0], max_range[1])
        image[:] = ((image / 255) ** rng) * 255
        
# AUGMENT ALL HSV VALUES
def augment_hsv(image, max_chance=0.5, max_range=0.3, bgr_order=True):
    if np.random.rand() > max_chance or image.ndim != 3 or image.shape[-1] < 3:
        return   
    
    r = np.random.uniform(-1, 1, 3) * [max_range, max_range, max_range] + 1 
    hue, sat, val = cv2.split(cv2.cvtColor(image[...,:3], cv2.COLOR_BGR2HSV if bgr_order else cv2.COLOR_RGB2HSV))
    dtype = image.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    image[...,:3] = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR if bgr_order else cv2.COLOR_HSV2RGB)
    
# JPEG COMPRESSION ARTIFACTS
def augment_compress_artifacts(image, max_chance=0.5, max_range=0.4):
    # Example, `max_range = 0.25` means compression quality "75/100"
    if np.random.rand() > max_chance:
        return 
    
    rate = int(100 - np.random.uniform(0, max_range) * 100)    
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), rate]
    image[:] = cv2.imdecode(cv2.imencode('.jpg', image, encode_param)[1], -1)

# ADJUST COLOR HUE (needs a BGR input)
def augment_colorize(image, max_chance=0.5, max_range=0.1):
    if np.random.rand() < max_chance:
        rng = np.random.uniform(-180*max_range, 180*max_range)
        hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2HSV)
        hsv[:,:,0] = np.clip( hsv[:,:,0] + rng, 0, 179 )
        image[:,:,:3] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)