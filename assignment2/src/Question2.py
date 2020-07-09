import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu
from skimage.color import label2rgb
from skimage.color import rgb2gray
import skimage.morphology as m
from skimage import measure
from random import random
import matplotlib

def FindPixel(img): #find foreground pixel image
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > 0:
                img2 = np.zeros((img.shape[0], img.shape[1]))
                img2[i][j] = img[i][j]
                return img2
    return np.zeros(1)

def connectedExtraction (img, imgE):
    SE = m.square(3)
    imgO = np.logical_and(m.dilation(imgE, SE), img)
    imgTemp = np.ones(1)
    while(not (imgO==imgTemp).all()): #check for change in subsequent iterations
        imgTemp = np.logical_and(m.dilation(imgO, SE), img)
        imgO = np.logical_and(m.dilation(imgTemp, SE), img)
    return imgO

def ExtractAndLabelComponents(img):
    components = []
    imgP = FindPixel(img)
    while imgP.shape[0] != 1: #extract each component
        components.append(connectedExtraction(img, imgP))
        img = np.bitwise_xor(img,components[len(components)-1])
        imgP = FindPixel(img)
    for i in range(len(components)): #label
        components[i] = np.multiply(components[i], i+1)
    return components

colors = [(0,0,0)] #generate colour mapping
for i in range(255):
    if i%25==0 and i >= 50:
        colors.append((i/255,random(),random()))
for i in range(255):
    if i%25==0 and i >= 50:
        colors.append((random(),i/255,random()))
for i in range(255):
    if i%25==0 and i >= 50:
        colors.append((random(),random(),i/255))
new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=27)

img2 = rgb2gray(plt.imread('imgs/DIP.tif'))
thresh = threshold_otsu(img2)
binary = rgb2gray(img2 > thresh)
c = ExtractAndLabelComponents(binary)
l = c[0]
for i in range(len(c)-1):
    l += c[i+1]
print("Number of connected components in DIP: ", len(c)) #number of connected components
comparison = measure.label(binary)
plt.gray()
plt.imsave("output/2b.jpg",l, cmap=new_map)

img22 = rgb2gray(plt.imread('imgs/balls-with-reflections.tif'))
thresh2 = threshold_otsu(img22)
binary2 = rgb2gray(img22 > thresh2)
c2 = ExtractAndLabelComponents(binary2)
l2 = c2[0]
for i in range(len(c2)-1):
    l2 += c2[i+1]
print("Number of connected components in balls-with-reflections: ", len(c2)) #number of connected components
comparison2 = measure.label(binary2)
plt.gray()
plt.imsave("output/2b-2.jpg",l2, cmap=new_map)

fig, axs = plt.subplots(1,2,figsize=(9,5))
[axi.set_axis_off() for axi in axs.ravel()]
axs[0] = fig.add_subplot(1,2,1)
axs[0].set_title("Original")
axs[0].imshow(img2)
plt.axis('off')
axs[1] = fig.add_subplot(1,2,2)
axs[1].set_title("Coloured")
axs[1].imshow(l,  cmap = new_map)
plt.axis('off')
fig.savefig("output/2bComparison.jpg")

fig, axs = plt.subplots(1,2,figsize=(9,5))
[axi.set_axis_off() for axi in axs.ravel()]
axs[0] = fig.add_subplot(1,2,1)
axs[0].set_title("Original")
axs[0].imshow(img22)
plt.axis('off')
axs[1] = fig.add_subplot(1,2,2)
axs[1].set_title("Coloured")
axs[1].imshow(l2,  cmap = new_map)
plt.axis('off')
fig.savefig("output/2b-2Comparison.jpg")

fig, axs = plt.subplots(1,2,figsize=(9,5))
[axi.set_axis_off() for axi in axs.ravel()]
axs[0] = fig.add_subplot(1,2,1)
axs[0].set_title("Implementation version")
axs[0].imshow(l,  cmap = new_map)
plt.axis('off')
axs[1] = fig.add_subplot(1,2,2)
axs[1].set_title("skimage version")
axs[1].imshow(comparison,  cmap = new_map)
plt.axis('off')
fig.savefig("output/2c.jpg")

fig, axs = plt.subplots(1,2,figsize=(9,5))
[axi.set_axis_off() for axi in axs.ravel()]
axs[0] = fig.add_subplot(1,2,1)
axs[0].set_title("Implementation version")
axs[0].imshow(l2,  cmap = new_map)
plt.axis('off')
axs[1] = fig.add_subplot(1,2,2)
axs[1].set_title("skimage version")
axs[1].imshow(comparison2,  cmap = new_map)
plt.axis('off')
fig.savefig("output/2c-2.jpg")
