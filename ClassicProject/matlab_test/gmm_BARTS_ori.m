%% Test function for gmm of BARTS dataset for brain tumor segmentaiton:
% 29.11.17 (c) Roy Hirsch

clear all;

%% load data
% load the image matrix named Im
load('\FinalProject\BRATS_HG0001\dataBN.mat')
% load the label matrix, named gt4
load('\FinalProject\BRATS_HG0001\gt4.mat')
im = double(im);

%%
%T2
mod = 2;
sliceZ = 70;
backThs = exp(-4);


%% regular gmm on a single modality
% select and pre-prepare a modality for gmm:
img = im(:,:,:,mod);
figure;
imshow(img(:,:,sliceZ)/(2^11));
maxMatrix = max(img(:));
im_n = img / maxMatrix;
im_n(im_n<backThs) = 0;
im_n_vec = im_n(im_n~=0);


%% 3. Train gmm with bakround voxal and regularization
gm = fitgmdist(im_n(:),5,'RegularizationValue',0.003,'Options',statset('MaxIter',400));
pdfM2 = createPDFMatrix(gm, im_n, 1);
out2 = createSegmentetionMatrix(pdfM2,1,70);

%%
%flair
mod=4;
%% regular gmm on a single modality
% select and pre-prepare a modality for gmm:
img = im(:,:,:,mod);
figure;
imshow(img(:,:,sliceZ)/(2^11));
maxMatrix = max(img(:));
im_n = img / maxMatrix;
im_n(im_n<backThs) = 0;
im_n_vec = im_n(im_n~=0);
figure;
imshow(im_n);

%% 3. Train gmm with bakround voxal and regularization
a=im_n(:);
gm = fitgmdist(a(a~=0),4,'RegularizationValue',0.005,'Options',statset('MaxIter',200));
pdfM4 = createPDFMatrix(gm, im_n, 1);
%%
out = createSegmentetionMatrix(pdfM4,1,70);
%%
figure;
imhist(im_n(:),256);


