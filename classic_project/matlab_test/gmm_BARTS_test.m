%% Test function for gmm of BARTS dataset for brain tumor segmentaiton:
% 29.11.17 (c) Roy Hirsch

clear all;

%% load data
% load the image matrix named Im
load('/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0001/dataBN.mat')
% load the label matrix, named gt4
load('/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0001/gt4.mat')
im = double(im);

% parameters:
mod = 3;
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

%% optional: pre-process :
%% image adjastment
im_a = zeros(size(im_n));
for i=1:length(im_n(1,1,:))
    temp = im_n(:,:,i);
    temp = imadjust(temp);
    im_a(:,:,i) = temp;
end

%% gamma correction
im_g = zeros(size(im_n));
for i=1:length(im_n(1,1,:))
    temp = im_n(:,:,i);
    temp = imadjust(temp,[],[],0.5);
    im_g(:,:,i) = temp;
end
% plot for compare
figure;subplot(1,3,1); imshow(im_n(:,:,60));title('original');
subplot(1,3,2); imshow(im_a(:,:,60));title('adjusted');
subplot(1,3,3); imshow(im_g(:,:,60));title('gamma correction');

figure;subplot(1,3,1); imshow(im_n(:,:,105));title('original');
subplot(1,3,2); imshow(im_a(:,:,105));title('adjusted');
subplot(1,3,3); imshow(im_g(:,:,105));title('gamma correction');
%% gmm
gm = fitgmdist(im_n_vec(:),5,'Options',statset('MaxIter',400));
pdfM = createPDFMatrix(gm, im_n, 1);
%% print segmentation
out = createSegmentetionMatrix(pdfM,1,90);
%
%
%
%
%
%
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
%im_n = loadMRImage('/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0004/dataBN.mat',2,exp(-4),0);
%%
gm = fitgmdist(im_a(:),5,'RegularizationValue',0.003,'Options',statset('MaxIter',400));
pdfM = createPDFMatrix(gm, im_a, 0);
segMatrix = createSegmentetionMatrix(pdfM,1,80);
%%
figure; imshow(segMatrix(:,:,60)/5);
%figure; imshow(im_n(:,:,105));
%% 4. generating few GMM models for each modality
% Generate four different vectors for each modality to train GMM model:
im_n = zeros(size(im));
for i=1:4
    temp = im(:,:,:,i);
    maxMatrix = max(temp(:));
    im_n(:,:,:,i) = im(:,:,:,i)/maxMatrix;
    im_n(im_n<backThs) = 0;
end
im_T1_vec = im_n(:,:,:,1);
im_T2_vec = im_n(:,:,:,2);
im_T1t_vec = im_n(:,:,:,3);
%im_FL_vec = im_n(:,:,:,4);

%% image adjastment to FL
im_a = zeros(size(im_n(:,:,:,4)));
for i=1:length(im_n(1,1,:,1))
    temp = im_n(:,:,i);
    temp = imadjust(temp);
    im_a(:,:,i) = temp;
end
im_FL_vec = im_a;
%% gmm train
% gmm of T1
% gmT1 = fitgmdist(im_T1_vec,3,'Options',statset('MaxIter',400));
% pdfMT2 = createPDFMatrix(gmT1, im_n(:,:,:,1), 0);
% createSegmentetionMatrix(pdfMT2,1,80);

% gmm of T2
gmT2 = fitgmdist(im_T2_vec(:),5,'RegularizationValue',0.003,'Options',statset('MaxIter',200));
pdfMT2 = createPDFMatrix(gmT2, im_n(:,:,:,2), 0);
segMatrixT2 = createSegmentetionMatrix(pdfMT2,1,80);

%gmm of T1t
gmT1t = fitgmdist(im_T1t_vec(:),5,'RegularizationValue',0.003,'Options',statset('MaxIter',200));
pdfMT1t = createPDFMatrix(gmT1t, im_n(:,:,:,3), 0);
createSegmentetionMatrix(pdfMT1t,1,80);

%% gmm of FLAIR
gmFL = fitgmdist(im_FL_vec(:),4,'RegularizationValue',0.007,'Options',statset('MaxIter',200));
pdfMFL = createPDFMatrix(gmFL, im_n(:,:,:,4), 0);
segMatrixFL = createSegmentetionMatrix(pdfMFL,1,80);


%%
%
%
% Accuracy 
% create the prediction mask of T2 and FL
predictT2 = zeros(size(segMatrixT2));
predictFL = zeros(size(segMatrixT2));

predictT2(segMatrixT2 == 2 | segMatrixT2 == 5) = 1;
predictFL(segMatrixFL == 3) = 1;
predict = and(predictT2,predictFL);

% calculate dice score
labels = double(gt4);
labels(labels~=0) = 1;
dice =  2*sum(sum(sum(and(labels,predict)))) / sum(sum(sum((labels + predict))));
%% print label
curve = gradient(predict);
figure;subplot(1,2,1);imshow(1-curve(:,:,100));
subplot(1,2,2);imshow(im_n(:,:,100));
%% help:
figure; subplot(2,2,1); imshow(im_n(:,:,100,1));title('T1');
subplot(2,2,2); imshow(im_n(:,:,100,2));title('T2');
subplot(2,2,3); imshow(im_n(:,:,100,3));title('T1t');
subplot(2,2,4); imshow(im_n(:,:,100,4));title('FL');