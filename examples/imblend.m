function [ w, wmax, wmin ] = imblend( f, g )
%IMBLEND Weighted sum of two images
%   [W, WMAX, WMIN] = IMBLEND(F, G) computes a weighted sum (W) of 
%   two input images, F and G. IMBLEND also computes the maximum (WMAX) and
%   minimum (WMIN) values of W. F and G must be of the same size and
%   numeric class. The output image is of the same class as the input
%   images.
w1 = 0.5 * f;
w2 = 0.5 * g;
w = w1 + w2;

wmax = max(w(:));
wmin = min(w(:));
end

