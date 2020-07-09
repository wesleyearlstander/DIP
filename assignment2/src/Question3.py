import matplotlib.pyplot as plt
import skimage.morphology as m
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
import numpy as np

img3 = rgb2gray(plt.imread('imgs/blobs.tif'))
SE = m.disk(28)
img31 = m.closing(img3, SE) #remove small blobs
SE2 = m.disk(50)
img32 = m.opening(img31, SE2) #get big side
thresh3 = threshold_otsu(img32)
imgBoundary = img32 - m.erosion(img32, m.disk(5)) > thresh3 #extract boundary
output = np.multiply(imgBoundary,255)+img3
plt.gray()
plt.imsave("output/3.png", output, vmin=0, vmax=255)

fig, axs = plt.subplots(1,2,figsize=(9,5))
[axi.set_axis_off() for axi in axs.ravel()]
axs[0] = fig.add_subplot(1,2,1)
axs[0].set_title("Original")
axs[0].imshow(img3,  vmin = 0, vmax= 255)
plt.axis('off')
axs[1] = fig.add_subplot(1,2,2)
axs[1].set_title("Seperated")
axs[1].imshow(output,  vmin = 0, vmax= 255)
plt.axis('off')
fig.savefig("output/3Comparison.jpg")
