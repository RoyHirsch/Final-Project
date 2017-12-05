%% Diffrent gmm

clear all;
%% load data
% load the image matrix named Im
load('/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0001/dataBN.mat')
% load the label matrix, named gt4
% load('/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0001/gt4.mat')
im = double(im);

% parameters:
mod = 3;
sliceZ = 80;
backThs = exp(-4);
%

% select and pre-prepare a modality for gmm:
img = im(:,:,:,mod);
maxMatrix = max(img(:));
im_n = img / maxMatrix;
im_n(im_n<backThs) = 0;
im_n_vec = im_n(im_n~=0);

%%
im_g = imgaussfilt3(im_n);
im_g_vec = im_g(im_g~=0);
%%

gm = fitgmdist(im_g_vec(:),3,'RegularizationValue',0.001,'Options',statset('MaxIter',400));
pdfM = createPDFMatrix(gm, im_g, 0);
segMatrix = createSegmentetionMatrix(pdfM,1,80);
%%
%WDEC = wavedec3(im_n,3,'haar');
WT = dwt3(im_n,'haar');
val = WT.dec;
val2 = val(1,1);
%[H,V,D] = detcoef2('all',C,S,1);
%ALL = H + V + D;
%figure; imshow(ALL(:,:,80));
%%
figure;subplot(2,2,1);imshow(H/max(max((H))));title(['Horizontal coefficients, level: ',num2str(level)]);
subplot(2,2,2);imshow(H/max(max((V))));title(['Vertical coefficients, level: ',num2str(level)]);
subplot(2,2,3);imshow(H/max(max((D))));title(['Diagonal coefficients, level: ',num2str(level)]);