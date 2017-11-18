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

%% GMM of T1 channel
% a tryout to estimate a geussian model over the data

clear all;
load('dataBN.mat');
img = double(im);
[H W D mod] = size(img);
img_T1 = img(:,:,:,1);

max_T1 = max(img_T1(:));
img_T1 = img_T1/max_T1;
test = img_T1(:,:,100);
[counts,binLocations] = imhist(img_T1(img_T1>0.1),256);
max_count = max(counts);
counts = counts / max_count;
gm = fitgmdist([binLocations,counts],3);
figure;
bar(binLocations,counts)
xlim = ([0 1]);

hold on;
x = 0:1/256:1
n1 = makedist('normal',gm.mu(1),sqrt(gm.Sigma(1,1,1)));
n2 = makedist('normal',gm.mu(2),sqrt(gm.Sigma(1,1,2)));
n3 = makedist('normal',gm.mu(3),sqrt(gm.Sigma(1,1,3)));
pdf_n1 = pdf(n1,x);
pdf_n1 = pdf_n1 / max(pdf_n1);
plot(x,pdf_n1);

pdf_n2 = pdf(n2,x);
pdf_n2 = pdf_n2 / max(pdf_n2);
plot(x,pdf_n2);

pdf_n3 = pdf(n3,x);
pdf_n3 = pdf_n3 / max(pdf_n3);
plot(x,pdf_n3);
% 
% n4 = makedist('normal',gm.mu(4),sqrt(gm.Sigma(1,1,4)));
% pdf_n4 = pdf(n4,x);
% pdf_n4 = pdf_n4 / max(pdf_n4);
% plot(x,pdf_n4);
hold off;
%ezcontour(@(x,y)pdf(gm,[x y]),[0 1],[0 1])
