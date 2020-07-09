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

def histEqual(f):
    out = np.copy(f)
    rk, nk = np.unique(out, return_counts=True) #unique intensities and respective counts
    pk = nk/float(np.size(f)) #probability of each unique intensity
    sk = np.cumsum(pk) #cumulative sums of pixel probabilities
    mul = sk*255.0 #cumulative frequency multiplied by max value
    roundVal = np.round(mul) #round each multiple
    for i in range(len(f)):
        for j in range(len(f[0])):
            out[i][j] = roundVal[np.where(rk==f[i][j])] #map pixels for equalization
    return out

hist11 = histEqual(img1)
print("Equalize Hist 1")

hist21 = histEqual(img2)
print("Equalize Hist 2")

img311 = ScaleIntensity(img3, contrast(img3)*255) #image must be scaled as it is 0-1

hist31 = histEqual(img311)
print("Equalize Hist 3")

def AdaptHistEqual(f, blockSize = 16):
    out = np.copy(f)
    padAmount = int(round(blockSize / 2)) #amount to pad at each edge for block to be used
    padded = np.pad(out, pad_width=((padAmount, padAmount), (padAmount, padAmount)), mode='symmetric') #pad array
    for i in range(len(f)):
        for j in range(len(f[0])):
            block = padded[i:i+blockSize,j:j+blockSize] #extract square for each pixel
            rk, nk = np.unique(block, return_counts=True) #unique intensities and respective counts
            pk = nk/float(np.size(block)) #probability of each unique intensity
            sk = np.cumsum(pk) #cumulative sums of pixel probabilities
            mul = sk*255.0 #cumulative frequency multiplied by max value
            roundVal = np.round(mul) #round each multiple
            out[i][j] = roundVal[np.where(rk==block[padAmount-1][padAmount-1])] #extract central pixel value
    return out

hist12 = AdaptHistEqual(img1,64)
print("adaptHist 1")
hist22 = AdaptHistEqual(img2,64)
print("adaptHist 2")
hist32 = AdaptHistEqual(img311,64)
print("adaptHist 3")

fig, axs = plt.subplots(3,2,figsize=(12,10))
[axi.set_axis_off() for axi in axs.ravel()]
axs[0][0] = fig.add_subplot(3,2,1)
axs[0][0].set_title("Original Image")
axs[0][0].imshow(img1,  vmin = 0, vmax= 255)
plt.axis('off')
axs[1][0] = fig.add_subplot(3,2,3)
axs[1][0].set_title("Equalized Histogram Image")
axs[1][0].imshow(hist11,  vmin = 0, vmax= 255)
plt.axis('off')
axs[2][0] = fig.add_subplot(3,2,5)
axs[2][0].set_title("Adaptive Histogram Image")
axs[2][0].imshow(hist12,  vmin = 0, vmax= 255)
plt.axis('off')
axs[0][1] = fig.add_subplot(3,2,2)
axs[0][1].set_title("Original Histogram")
axs[0][1].hist(img1.ravel(), 256, [0,256], density=1)
axs[1][1] = fig.add_subplot(3,2,4)
axs[1][1].set_title("Equalized Histogram")
axs[1][1].hist(hist11.ravel(), 256, [0,256], density=1)
axs[2][1] = fig.add_subplot(3,2,6)
axs[2][1].set_title("Adaptive Histogram")
axs[2][1].hist(hist12.ravel(), 256, [0,256], density=1)
fig.savefig("output/Road1_1-2+3.jpg")

fig, axs = plt.subplots(3,2,figsize=(12,10))
[axi.set_axis_off() for axi in axs.ravel()]
axs[0][0] = fig.add_subplot(3,2,1)
axs[0][0].set_title("Original Image")
axs[0][0].imshow(img2,  vmin = 0, vmax= 255)
plt.axis('off')
axs[1][0] = fig.add_subplot(3,2,3)
axs[1][0].set_title("Equalized Histogram Image")
axs[1][0].imshow(hist21,  vmin = 0, vmax= 255)
plt.axis('off')
axs[2][0] = fig.add_subplot(3,2,5)
axs[2][0].set_title("Adaptive Histogram Image")
axs[2][0].imshow(hist22,  vmin = 0, vmax= 255)
plt.axis('off')
axs[0][1] = fig.add_subplot(3,2,2)
axs[0][1].set_title("Original Histogram")
axs[0][1].hist(img2.ravel(), 256, [0,256], density=1)
axs[1][1] = fig.add_subplot(3,2,4)
axs[1][1].set_title("Equalized Histogram")
axs[1][1].hist(hist21.ravel(), 256, [0,256], density=1)
axs[2][1] = fig.add_subplot(3,2,6)
axs[2][1].set_title("Adaptive Histogram")
axs[2][1].hist(hist22.ravel(), 256, [0,256], density=1)
fig.savefig("output/Road2_1-2+3.jpg")

fig, axs = plt.subplots(3,2,figsize=(12,10))
[axi.set_axis_off() for axi in axs.ravel()]
axs[0][0] = fig.add_subplot(3,2,1)
axs[0][0].set_title("Original Image")
axs[0][0].imshow(img3,  vmin = 0, vmax= 1)
plt.axis('off')
axs[1][0] = fig.add_subplot(3,2,3)
axs[1][0].set_title("Equalized Histogram Image")
axs[1][0].imshow(hist31,  vmin = 0, vmax= 255)
plt.axis('off')
axs[2][0] = fig.add_subplot(3,2,5)
axs[2][0].set_title("Adaptive Histogram Image")
axs[2][0].imshow(hist32,  vmin = 0, vmax= 255)
plt.axis('off')
axs[0][1] = fig.add_subplot(3,2,2)
axs[0][1].set_title("Original Histogram")
axs[0][1].hist(img311.ravel(), 256, [0,256], density=1)
axs[1][1] = fig.add_subplot(3,2,4)
axs[1][1].set_title("Equalized Histogram")
axs[1][1].hist(hist31.ravel(), 256, [0,256], density=1)
axs[2][1] = fig.add_subplot(3,2,6)
axs[2][1].set_title("Adaptive Histogram")
axs[2][1].hist(hist32.ravel(), 256, [0,256], density=1)
fig.savefig("output/Sports_1-2+3.png")


