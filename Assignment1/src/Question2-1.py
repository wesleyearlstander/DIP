import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from scipy import fftpack as ff
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

def paddedSize(img):
    return (2*img.shape[0])-1, (2*img.shape[1])-1

def freqz2(P, Q, T, param, w=1, order=3):
    u = np.arange(0,P,1.0)
    v = np.arange(0,Q,1.0)
    idx = np.where(u>P/2)
    u[idx] = u[idx] - P
    idy = np.where(v>Q/2)
    v[idy] = v[idy]-Q
    V,U = np.meshgrid(v,u)
    D = (V**2 + U**2)**(1/2)
    out = np.zeros((P, Q))
    if (T == "ILPF"):
        for i in range(P):
            for j in range(Q):
                if (D[i][j] <= param):
                    out[i][j] = 1
    elif (T == "GLPF"):
        for i in range(P):
            for j in range(Q):
                out[i][j] = math.exp((-1*(D[i][j]**2))/(2*(param**2)))
    elif (T == "BLPF"):
        for i in range(P):
            for j in range(Q):
                out[i][j] = 1/(1+(D[i][j]/param)**(2*order)) #butter-worth low pass filter
    elif (T == "BRBF"):
        for i in range(P):
            for j in range(Q):
                out[i][j] = 1 / (1+ ((w * D[i][j])/((D[i][j])**2 - param**2))**(2*order)) #n=3
    return out

def filterImageButterWorth (img, amount, order =3):
    p = paddedSize(img)
    originalSize = [img.shape[0], img.shape[1]]
    img = np.pad(img, ((0, p[0]-originalSize[0]), (0, p[1]-originalSize[1])))
    imgDFT = ff.fft2(img)
    H = freqz2(p[0], p[1], "BLPF", amount, 1, order)
    imgFiltered = np.multiply(imgDFT, H)
    imgOut = ff.ifft2(imgFiltered)[0:originalSize[0], 0:originalSize[1]]
    return np.real(imgOut)

def UnsharpMaskButterWorth (img, amount, order=3):
    p = paddedSize(img)
    originalSize = [img.shape[0], img.shape[1]]
    img = np.pad(img, ((0, p[0]-originalSize[0]), (0, p[1]-originalSize[1])))
    imgDFT = ff.fft2(img)
    H = freqz2(p[0], p[1], "BLPF", amount, 1, order)
    H = np.subtract(1,H)
    imgFiltered = np.multiply(imgDFT, H)
    imgOut = ff.ifft2(imgFiltered)[0:originalSize[0], 0:originalSize[1]]
    return np.real(imgOut)

def BandRejectButterWorth (img, amount, w, order=3):
    p = paddedSize(img)
    originalSize = [img.shape[0], img.shape[1]]
    img = np.pad(img, ((0, p[0]-originalSize[0]), (0, p[1]-originalSize[1])))
    imgDFT = ff.fft2(img)
    H = freqz2(p[0], p[1], "BRBF", amount, w, order)
    imgFiltered = np.multiply(imgDFT, H)
    imgOut = ff.ifft2(imgFiltered)[0:originalSize[0], 0:originalSize[1]]
    return np.real(imgOut)

def BandPassButterWorth (img, amount, w, order=3):
    p = paddedSize(img)
    originalSize = [img.shape[0], img.shape[1]]
    img = np.pad(img, ((0, p[0]-originalSize[0]), (0, p[1]-originalSize[1])))
    imgDFT = ff.fft2(img)
    H = freqz2(p[0], p[1], "BRBF", amount, w, order)
    H = np.subtract(1,H)
    imgFiltered = np.multiply(imgDFT, H)
    imgOut = ff.ifft2(imgFiltered)[0:originalSize[0], 0:originalSize[1]]
    return np.real(imgOut)

imgZ1 = filterImageButterWorth(imgZ,57, 6)

imgZ2 = UnsharpMaskButterWorth(imgZ, 57, 6)

imgZ3 = ScaleIntensity(imgZ2,255)

imgZ4 = BandRejectButterWorth(imgZ, 53, 37, 3)

imgZ5 = BandPassButterWorth(imgZ, 50, 37, 2)

imgZ6 = ScaleIntensity(imgZ5,255)

fig, axs = plt.subplots(2,3,figsize=(12,8))
[axi.set_axis_off() for axi in axs.ravel()]
axs[0][0] = fig.add_subplot(2,3,1)
axs[0][0].set_title("Lowpass Butterworth Filter")
axs[0][0].imshow(imgZ1,  vmin = 0, vmax= 255)
plt.axis('off')
axs[1][0] = fig.add_subplot(2,3,2)
axs[1][0].set_title("Highpass Butterworth Filter")
axs[1][0].imshow(imgZ2, vmin=0, vmax=255)
plt.axis('off')
axs[0][1] = fig.add_subplot(2,3,3)
axs[0][1].set_title("Rescaled Highpass Filter")
axs[0][1].imshow(imgZ3)
plt.axis('off')
axs[1][1] = fig.add_subplot(2,3,4)
axs[1][1].set_title("Bandreject Butterworth Filter")
axs[1][1].imshow(imgZ4, vmin=1, vmax=255)
plt.axis('off')
axs[0][2] = fig.add_subplot(2,3,5)
axs[0][2].set_title("Bandpass Butterworth Filter")
axs[0][2].imshow(imgZ5, vmin=0, vmax=255)
plt.axis('off')
axs[1][2] = fig.add_subplot(2,3,6)
axs[1][2].set_title("Rescaled Bandpass Filter")
axs[1][2].imshow(imgZ6)
plt.axis('off')
fig.savefig("imgs/2-1.png", vmin=0, vmax=255)
