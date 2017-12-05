%% Test function for gmm of BARTS dataset for brain tumor segmentaiton:
% 29.11.17 (c) Roy Hirsch
% matObj = matfile('dataBN.mat');

clear all;
addpath(genpath('/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc'))

% Load all data and labels:
data = load_all_data();
labels = load_all_labels();

%% Load single image:
% load the image matrix named Im
load('/Data/BRATS_HG0001/dataBN.mat','im')
% load the label matrix, named gt4
load('/Data/BRATS_HG0001/gt4.mat')
labales 
%% Parameters:
params.mod = 3;
params.sliceZ = 80;
params.backThs = exp(-4);

%% Regular gmm on a single modality
% Select and pre-prepare a modality for gmm:
img = im(:,:,:,params.mod);
maxMatrix = max(img(:));
im_n = img / maxMatrix;
im_n(im_n<params.backThs) = 0;
im_n_vec = im_n(im_n~=0);

%% Train gmm with bakround voxal and regularization
%im_n = loadMRImage('/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0004/dataBN.mat',2,exp(-4),0);
gm = fitgmdist(im_a(:),5,'RegularizationValue',0.003,'Options',statset('MaxIter',400));
pdfM = createPDFMatrix(gm, im_a, 0);
segMatrix = createSegmentetionMatrix(pdfM,1,80);
figure; imshow(segMatrix(:,:,60)/5);
%figure; imshow(im_n(:,:,105));
%% 4. generating few GMM models for each modality
% Generate four different vectors for each modality to train GMM model:
im_n = zeros(size(im));
for i=1:4
    temp = im(:,:,:,i);
    maxMatrix = max(temp(:));
    im_n(:,:,:,i) = im(:,:,:,i)/maxMatrix;
    im_n(im_n<params.backThs) = 0;
    % for pre-processing of FL 
    if i == 4
       imFL = im_n(:,:,:,4);
       im_a = zeros(size(imFL));
       for i=1:length(imFL(1,1,:))
           temp = imFL(:,:,i);
           temp = imadjust(temp);
           im_a(:,:,i) = temp;
       end
       im_n(:,:,:,4) = im_a;
    end
end
im_T1_vec = im_n(:,:,:,1);
im_T2_vec = im_n(:,:,:,2);
im_T1t_vec = im_n(:,:,:,3);
im_FL_vec = im_n(:,:,:,4);

%% gmm train
% gmm of T1
% gmT1 = fitgmdist(im_T1_vec,3,'Options',statset('MaxIter',400));
% pdfMT2 = createPDFMatrix(gmT1, im_n(:,:,:,1), 0);
% createSegmentetionMatrix(pdfMT2,1,80);

% gmm of T2
gmT2 = fitgmdist(im_T2_vec(:),5,'RegularizationValue',0.003,'Options',statset('MaxIter',400));
pdfMT2 = createPDFMatrix(gmT2, im_n(:,:,:,2), 0);
segMatrixT2 = createSegmentetionMatrix(pdfMT2,1,80);

%gmm of T1t
% gmT1t = fitgmdist(im_T1t_vec(:),5,'RegularizationValue',0.003,'Options',statset('MaxIter',200));
% pdfMT1t = createPDFMatrix(gmT1t, im_n(:,:,:,3), 0);
% createSegmentetionMatrix(pdfMT1t,1,80);

% gmm of FLAIR
gmFL = fitgmdist(im_FL_vec(:),4,'RegularizationValue',0.005,'Options',statset('MaxIter',200));
pdfMFL = createPDFMatrix(gmFL, im_n(:,:,:,4), 0);
segMatrixFL = createSegmentetionMatrix(pdfMFL,1,80);


%% Accuracy 
% Create the prediction mask of T2 and FL
predictT2 = zeros(size(segMatrixT2));
predictFL = zeros(size(segMatrixT2));

predictT2(segMatrixT2 == 1 | segMatrixT2 == 5) = 1;
predictFL(segMatrixFL == 3) = 1;
predict = and(predictT2,predictFL);
% Figure
figure; subplot(1,3,1); imshow(predictT2(:,:,80));subplot(1,3,2); imshow(predictFL(:,:,80));subplot(1,3,3); imshow(predict(:,:,80));
% Calculate dice score
labels = double(gt4);
% figure; imshow(labels(:,:,80)/4);
labels(labels~=0) = 1;
print_boundary(predict,im_n,params);
dice =  dice(labels,predict);

%% help:
figure; subplot(2,2,1); imshow(im_n(:,:,100,1));title('T1');
subplot(2,2,2); imshow(im_n(:,:,100,2));title('T2');
subplot(2,2,3); imshow(im_n(:,:,100,3));title('T1t');
subplot(2,2,4); imshow(im_n(:,:,100,4));title('FL');
