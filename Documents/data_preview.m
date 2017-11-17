% Experience for opening a sample from the BARTS dataset
% locate the script in folder with the .mat failes.
clear all;

% load the data structes:
load('dataBN.mat');
load('gt4.mat');

% parameter
slice = 90;

% 'gt4' contains the labeling of the data, in range: [0,2,1,4]
% I'll divide the matrix by 4 to visualize the different tisuess with 'imread'
labeled_data = double(gt4(:,:,slice))/4;
imshow(labeled_data)


%% 'im' is a 4D matrix [H,W,D,modulation]
% modulation order = T1, T2, T1c, FLAIR
% the folowing program normalizes and visoalizes the data modulations

img = double(im);
[H W D mod] = size(img);
img_T1 = img(:,:,:,1);
img_T2 = img(:,:,:,2);
img_T1c = img(:,:,:,3);
img_FLAIR = img(:,:,:,4);

figure;

subplot(2,2,1);
img_T1_slice = img_T1(:,:,90);
max_t1 = max(img_T1_slice(:));
img_T1_slice = img_T1_slice/max_t1;
imshow(img_T1_slice)
title('T1 slice');

subplot(2,2,2);
img_T2_slice = img_T2(:,:,slice);
max_t2 = max(img_T2_slice(:));
img_T2_slice = img_T2_slice/max_t2;
imshow(img_T2_slice)
title('T2 slice');

subplot(2,2,3);
img_T1c_slice = img_T1c(:,:,slice);
max_t1c = max(img_T1c_slice(:));
img_T1c_slice = img_T1c_slice/max_t1c;
imshow(img_T1c_slice)
title('T1c slice');

subplot(2,2,4);
img_FLAIR_slice = img_FLAIR(:,:,slice);
max_flair = max(img_FLAIR_slice(:));
img_FLAIR_slice = img_FLAIR_slice/max_flair;
imshow(img_FLAIR_slice)
title('FLAIR slice');

