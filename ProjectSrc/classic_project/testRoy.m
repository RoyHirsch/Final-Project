%% Using quantization for BARTS HG tumor segmentation:

% 1. Load the data

clear all;
addpath(genpath('/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc'))
addpath(genpath('/Users/royhirsch/Downloads/Chan-Vese'))
% load the image matrix named Im
load('/Data/BRATS_HG0001/dataBN.mat','im')
% load the label matrix, named gt4
load('/Data/BRATS_HG0001/gt4.mat')

label = double(gt4);
label(label~=0) = 1;

zSlice = 80;
im = double(im);
thr = exp(-4);

% Separate to modalities:
imT1 = im(:,:,:,1);
imT2 = im(:,:,:,2);
imT1g = im(:,:,:,3);
imFL = im(:,:,:,4);

% Quantization for single example:
[X, Y, Z] = size(imT2);
threshT2 = multithresh(imT2, 2);

quantImage = zeros(X, Y, Z);
for z=1:Z
    quantImage(:,:,z) = imquantize(imT2(:,:,z),threshT2);
end

threshFL = multithresh(imFL, 2);
quantImageFL = zeros(X, Y, Z);
for z=1:Z
    quantImageFL(:,:,z) = imquantize(imFL(:,:,z),threshFL);
end

% Ceate predict mask
quantImage(quantImage~=3) = 0;
quantImageFL(quantImageFL~=3) = 0;

% Calculate predict mask and dicescore:
predict = quantImage & quantImageFL;
predictClean = cleanSegmentaionMask(predict,30);
diceScore = dice(predict,label)

%% chenvese

imT2Normalize = imT2 / max(imT2(:));
imFLNormalize = imFL / max(imFL(:));

temp = chenvese(imFLNormalize(:,:,60),predictClean(:,:,60),200);
temp2 = activecontour(imFLNormalize(:,:,80),predictClean(:,:,80),200,'Chan-Vese');



%%
chenvesePredict = zeros(X, Y, Z);
for z=1:Z
    chenvesePredict = chenvese(imT2Normalize(:,:,z),predict(:,:,z),200);
end
