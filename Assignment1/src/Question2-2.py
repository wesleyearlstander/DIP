import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import morphology as m
import math

def contrast(image):
    return np.max(image) - np.min(image)

def ScaleIntensity (image, K=1):
    m = contrast(image)
    out = np.array(image).astype(float)
    offset = K*float(np.min(image)/m)
    for j in range(image.shape[1]):
        for i in range(image.shape[0]):
            out[i][j] =  K*(out[i][j]/m) - offset
    return out

imgZ = rgb2gray(plt.imread("imgs/zoneplate.tif"))
plt.gray()

def gaussKernel(m,sig,K=1):
    out = np.zeros((m,m))
    total = 0
    for i in range(m):
        for j in range(m):
            r = ((i-(m-1)/2)**2) + ((j-(m-1)/2)**2)
            out[i][j] = K * math.exp(-1 * (r/(2*(sig**2))))
            total += out[i][j]
    return out/total;        

H = m.disk(9)
H = np.logical_and(H, gaussKernel(19, 2)) #disk guassian filter to reduce corner artifacts in output image

def twodConv(f,w):
    m = len(w)
    n = len(w[0])
    if m == n:
        y = len(f)
        x = len(f[0])
        g = np.pad(f, int((m-1)/2), mode='constant')
        new = np.zeros((y,x))
        for i in range(y):
            for j in range(x):
                    new[i][j] = np.sum(g[i:i+m, j:j+m] * w)
        return new
    
imgZ7 = twodConv(imgZ, H)
imgZ7 = ScaleIntensity(imgZ7,255)

imgZ8 = imgZ - imgZ7

fig, axs = plt.subplots(1,2,figsize=(8,4))
[axi.set_axis_off() for axi in axs.ravel()]
axs[0] = fig.add_subplot(1,2,1)
axs[0].set_title("Lowpass Butterworth Filter")
axs[0].imshow(imgZ7,  vmin = 0, vmax= 255)
plt.axis('off')
axs[1] = fig.add_subplot(1,2,2)
axs[1].set_title("Highpass Butterworth Filter")
axs[1].imshow(imgZ8, vmin=37, vmax=255)
plt.axis('off')
fig.savefig("output/2-2.png", vmin=0, vmax=255)
