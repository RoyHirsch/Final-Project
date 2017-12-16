%% Using quantization for BARTS HG tumor segmentation:

% 1. Load the data

clear all;
addpath(genpath('/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc'))
% load the image matrix named Im
load('/Data/BRATS_HG0006/dataBN.mat','im')
% load the label matrix, named gt4
load('/Data/BRATS_HG0006/gt4.mat')

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
predictClean = cleanSegmentaionMask(predict,10);
predictClean = fillSegmentaionMask(predictClean);
diceScore = dice(predictClean,label);

%% Plot slices:

maxT1 = max(imT1(:));
maxT2 = max(imT2(:));
maxT1g = max(imT1g(:));
maxFL = max(imFL(:));

imT1 = imT1 ./ maxT1;
imT2 = imT2 ./ maxT2;
imT1g = imT1g ./ maxT1g;
imFL = imFL ./ maxFL;

imT1(imT1<=thr) = 0;
imT2(imT2<=thr) = 0;
imT1g(imT1g<=thr) = 0;
imFL(imFL<=thr) = 0;

figure;
subplot(2,2,1);
imshow(imT1(:,:,zSlice));
title(' T1 image:')
subplot(2,2,2);
imshow(imT2(:,:,zSlice));
title('T2 image:')
subplot(2,2,3);
imshow(imT1g(:,:,zSlice));
title('T1g image:')
subplot(2,2,4);
imshow(imFL(:,:,zSlice));
title('FL image:')

%% Plot histograms and thresholds of quantizations:

imT1_vec = imT1(imT1~=0);
imT2_vec = imT2(imT2~=0);
imT1g_vec = imT1g(imT1g~=0);
imFL_vec = imFL(imFL~=0);

threshT2 = threshT2 / maxT2;
threshFL = threshFL / maxFL;

figure;

subplot(2,2,1);
imhist(imT1_vec,256);
title('Histogram of normalized T1 image:')
subplot(2,2,2);
imhist(imT2_vec,256);hold on;
line([threshT2(1),threshT2(1)],ylim,'color','r');
line([threshT2(2),threshT2(2)],ylim,'color','r');
title('Histogram of normalized T2 image:')
subplot(2,2,3);
imhist(imT1g_vec,256);
title('Histogram of normalized T1g image:')
subplot(2,2,4);
imhist(imFL_vec,256);hold on;
line([threshFL(1),threshFL(1)],ylim,'color','r');
line([threshFL(2),threshFL(2)],ylim,'color','r');
title('Histogram of normalized FL image:')

%% Quantization over filtered images:

guassianImT2 = imgaussfilt(imT2,0.5,'FilterSize',15);
% guassianImFL = imgaussfilt(imFL,1);
stackImages = zeros(X,Y,Z,4);
stackImages(:,:,:,2) = guassianImT2;
stackImages(:,:,:,4) = imFL;
figure; imshow3D(guassianImT2);
[diceScore, predict] = quantizationT2andFLSegmentation(stackImages,gt4);

%% Evaloation on many examples:

data = load_all_data();
label = load_all_labels();

numOfExamples = 20;
diceScoreArray = zeros(numOfExamples, 1);
for i=1:numOfExamples
    [diceScoreArray(i), predict] = quantizationT2andFLSegmentation(data(i).f,label(i).f);
end
avarageDicScor = sum(diceScoreArray) / numOfExamples; 
