# -*- coding: utf-8 -*-

import numpy as np
import cv2
import sys
from math import sqrt
from math import cos
from math import pi
import matplotlib.pyplot as plt 

img = cv2.imread('ProjectImages/small.png')
plt.imshow(img)
N = img.shape[0]
img_reshaped = img.reshape(3, N, N)
CTM = np.zeros((N, N)) # Cosine Transform Matrix

# Generate CTM and its transpose:

for rownum in range(N):
    for colnum in range(N):
        if rownum == 0:
            CTM[rownum][colnum] = sqrt(1/N)
        else:
            CTM[rownum][colnum] = sqrt(2/N)*cos((2*colnum+1)*pi*rownum/(2*N))
            
CTMT = np.transpose(CTM)

# Apply CTM to image:

for color_channel in range(img.shape[2]):
    img_reshaped[color_channel] = CTM*img_reshaped[color_channel][:][:]*CTMT

new_img = img_reshaped.reshape(N, N, 3)


plt.imshow(new_img)


            
