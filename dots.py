import numpy as np
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt

def neighbors_mean(img, diam=3):
    mask = np.ones((diam,diam))
    mask[diam//2, diam//2] = 0
    return ndimage.generic_filter(img, np.nanmean, mode='constant', cval=np.nan, footprint=mask)

def get_neighbors(x,y,img,diam=1): return img[y-diam:y+diam, x-diam:x+diam]

def custom_filter(img, diam=3):
    print (img.shape)
    height, width = img.shape[:2]
    pad = (diam-1)//2
    img = cv2.copyMakeBorder(img, pad,pad,pad,pad, cv2.BORDER_REPLICATE)
    eps = 5
    output = np.zeros((height,width))

    for y in np.arange(pad,height+pad):
        for x in np.arange(pad, width+pad):
            data = get_neighbors(x,y,img,pad)
            loc_min = np.min(data) + eps
            data = np.where(data > loc_min, 255, data)
            output[y-pad, x-pad] = data[1,1]

    return output


orig = cv2.imread("dots.png")
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

nimg = neighbors_mean(gray,3)
maxed = ndimage.generic_filter(nimg, np.nanmax, size=3, mode='constant', cval=np.nan)
dots = custom_filter(maxed,9)

print (dots.shape)
counter = 0
np.savetxt('dots.txt',dots, fmt= '%d')
for i in dots:
    for j in i:
        if j != 255: counter += 1

print (counter)

fig1 = plt.subplot(2,1,1)
fig2 = plt.subplot(2,1,2)
fig1.imshow(gray, cmap='gray')
fig2.imshow(dots, cmap='gray')
plt.show()
