%Read images
img1 = imread('images/chap1&2/girl.tif');
img2 = imread('images/chap1&2/boy.tif');
img3_rgb = imread("images/chap1&2/lenna-RGB.tif");
img3_gray = rgb2gray(img3_rgb);
img4 = imread('images/chap1&2/chronometer.tif');

%Obtain the size of an mage
[M,N] = size(img4);

%Rescale an image
img4_rescaled = imresize(img4, 0.25);
[M_rescaled, N_rescaled] = size(img4_rescaled);

%Display an image
figure, imshow(img4);

%Display images side by side
fig1 = figure, subplot(1,2,1);
imshow(img4);
title("Chronometer-original");
subplot(1,2,2);
imshow(img4_rescaled);
title("Chronometer-rescaled");

%Save an image
imwrite(img4_rescaled,'chronometer_rescaled.png');

%Save the figure as an image
saveas(fig1, 'chronometers', 'png');

%Display images in multiple rows and columns
fig2 = figure, subplot(2,2,1);
imshow(img1);
title("Girl");

subplot(2,2,2);
imshow(img2);
title("Boy");

subplot(2,2,3);
imshow(img3_rgb);
title("Lenna-RGB");

subplot(2,2,4);
imshow(img3_gray);
title("Lenna-gray");

saveas(fig2, 'multiple_images.png');

%Type 'close all' to close all the figures. 









