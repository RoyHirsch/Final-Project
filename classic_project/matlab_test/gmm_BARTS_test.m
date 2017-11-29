%% Test function for gmm of BARTS dataset for brain tumor segmentaiton:
% 29.11.17 (c) Roy Hirsch

clear all;

%% load data
% load the image matrix named Im
load('/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0001-2/dataBN.mat')
% load the label matrix, named gt4
load('/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0001-2/gt4.mat')
im = double(im);

% parameters:
mod = 2;
sliceZ = 80;
backThs = exp(-4);
%

im = im(:,:,:,mod);
maxMatrix = max(im(:));
im_n = double(im)/maxMatrix;
im_n(im_n<backThs) = 0;
im_n_vec = im_n(im_n~=0);

% For slice segmentaion:
% im_T1_slice = im(:,:,sliceZ);
% maxVal = max(max(im_T1_slice));
% im_T1_slice = im_T1_slice / maxVal;
% figure(1);imshow(im_T1_slice); title('Image slice:')
% 
% %im_T1_slice(im_T1_slice<backThs) = 0;
%slice_vec = im_T1_slice(im_T1_slice~=0);

%% histogram
figure;
imhist(im_n_vec);
%% gmm train
gm = fitgmdist(im_n_vec,3,'Options',statset('MaxIter',400));
pdfM = createPDFMatrix(gm, im_n, 1);

