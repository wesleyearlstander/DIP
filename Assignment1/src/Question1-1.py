import numpy as np
import matplotlib.pyplot as plt

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

img1 = plt.imread("imgs/road_low_1.jpg")
img2 = plt.imread("imgs/road_low_2.jpg")
img3 = plt.imread("imgs/sports_low.png")
plt.gray()

img11 = ScaleIntensity(img1,255)

img21 = ScaleIntensity(img2,255)

img31 = ScaleIntensity(img3,255)

fig, axs = plt.subplots(1,2,figsize=(14,4))
[axi.set_axis_off() for axi in axs.ravel()]
axs[0] = fig.add_subplot(1,2,1)
axs[0].set_title("Original")
axs[0].imshow(img1,  vmin = 0, vmax= 255)
plt.axis('off')
axs[1] = fig.add_subplot(1,2,2)
axs[1].set_title("Stretched")
axs[1].imshow(img11,  vmin = 0, vmax= 255)
plt.axis('off')
fig.savefig("output/Road1_1-1.jpg")

fig, axs = plt.subplots(1,2,figsize=(12,5))
[axi.set_axis_off() for axi in axs.ravel()]
axs[0] = fig.add_subplot(1,2,1)
axs[0].set_title("Original")
axs[0].imshow(img2,  vmin = 0, vmax= 255)
plt.axis('off')
axs[1] = fig.add_subplot(1,2,2)
axs[1].set_title("Stretched")
axs[1].imshow(img21,  vmin = 0, vmax= 255)
plt.axis('off')
fig.savefig("output/Road2_1-1.jpg")

fig, axs = plt.subplots(1,2,figsize=(12,4))
[axi.set_axis_off() for axi in axs.ravel()]
axs[0] = fig.add_subplot(1,2,1)
axs[0].set_title("Original")
axs[0].imshow(img3,  vmin = 0, vmax= 1)
plt.axis('off')
axs[1] = fig.add_subplot(1,2,2)
axs[1].set_title("Stretched")
axs[1].imshow(img31,  vmin = 0, vmax= 255)
plt.axis('off')
fig.savefig("output/Sports_1-1.png")


