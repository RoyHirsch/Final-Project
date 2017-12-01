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
%% regular gmm on a single modality
% select and pre-prepare a modality for gmm:
img = im(:,:,:,mod);
maxMatrix = max(img(:));
im_n = img / maxMatrix;
im_n(im_n<backThs) = 0;
im_n_vec = im_n(im_n~=0);


%% gmm
gm = fitgmdist(im_n_vec(:),5,'Options',statset('MaxIter',400));
pdfM = createPDFMatrix(gm, im_n, 1);
%% print segmentation
out = createSegmentetionMatrix(pdfM,1,90);

%% 2. gmm of multimodal data
[H W D C] = size(im);
all_dim_mat = reshape(im,[H*W*D,C]);
all_dim_mat(all_dim_mat<backThs) = 0;

% gmm
gm = fitgmdist(all_dim_mat,4,'RegularizationValue',0.003,'Options',statset('MaxIter',400));
pdfM = createPDFMatrix(gm, im_n, 1);

% test
%figure; imshow(im_n(:,:,80))


%% 3. Train gmm with bakround voxal and regularization

Mu=[0.12,0.3,0.43];
Sigma(:,:,1)=0.01;
Sigma(:,:,2)=0.04;
Sigma(:,:,3)=0.02;
PComponents = [0.1,0.5,0.4];
%S = struct('mu',Mu,'Sigma',Sigma,'ComponentProportion',PComponents);

gm = fitgmdist(im_n(:),4,'RegularizationValue',0.003,'Options',statset('MaxIter',400));
pdfM = createPDFMatrix(gm, im_n, 0);
createSegmentetionMatrix(pdfM,1,80);

%% 4. generating few GMM models for each modality
% Generate four different vectors for each modality to train GMM model:
im_n = zeros(size(im));
for i=1:4
    temp = im(:,:,:,i);
    maxMatrix = max(temp(:));
    im_n(:,:,:,i) = im(:,:,:,i)/maxMatrix;
    im_n(im_n<backThs) = 0;
end
im_T1_vec = im_n((im_n(:,:,:,1)~=0));
im_T2_vec = im_n((im_n(:,:,:,2)~=0));
im_T1t_vec = im_n((im_n(:,:,:,3)~=0));
im_FL_vec = im_n((im_n(:,:,:,4)~=0));

%% gmm train
% gmm of T2
gmT2 = fitgmdist(im_n_vec,5,'Options',statset('Display','final','MaxIter',400));
pdfMT2 = createPDFMatrix(gmT2, im_n, 1);

%gmm of T1t
gmT1t = fitgmdist(im_T1t_vec,3,'Options',statset('MaxIter',400));
pdfMT1 = createPDFMatrix(gmT1t, im_n(:,:,:,3), 1);

%gmm of FLAIR
gmFL = fitgmdist(im_FL_vec,3,'Options',statset('MaxIter',400));
pdfMFL = createPDFMatrix(gmFL, im_n(:,:,:,4), 1);
