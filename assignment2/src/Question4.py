import matplotlib.pyplot as plt
import numpy as np
from random import random
import matplotlib
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
import skimage.morphology as m

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

img4 = rgb2gray(plt.imread('imgs/coins.png'))
SE = m.disk(25)
img41 = m.dilation(img4, SE)
thresh4 = threshold_otsu(img41)
binary4 = img41 < thresh4
c4 = ExtractAndLabelComponents(binary4)
l4 = c4[0]
for i in range(len(c4)-1):
    l4 += c4[i+1]
print("Coin amount is: ", len(c4))
plt.gray()
plt.imsave("output/4.png", l4, cmap=new_map)

fig, axs = plt.subplots(1,2,figsize=(9,5))
[axi.set_axis_off() for axi in axs.ravel()]
axs[0] = fig.add_subplot(1,2,1)
axs[0].set_title("Original")
axs[0].imshow(img4)
plt.axis('off')
axs[1] = fig.add_subplot(1,2,2)
axs[1].set_title("Coloured and Seperated")
axs[1].imshow(l4,  cmap = new_map)
plt.axis('off')
fig.savefig("output/4Comparison.jpg")
