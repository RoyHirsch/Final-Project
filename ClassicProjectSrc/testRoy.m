%% Using quantization for BARTS HG tumor segmentation:

% 1. Load the data

clear all;
addpath(genpath('/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc'))
% load the image matrix named Im
load('BRATS_HG0001/dataBN.mat','im')
% load the label matrix, named gt4
load('BRATS_HG0001/gt4.mat')

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

%% 
max1 = max(imT1(:));
imT1 = imT1 / max1;
imT1(imT1<thr) = 0;

max2 = max(imT2(:));
imT2 = imT2 / max2;
imT2(imT2<thr) = 0;

max3 = max(imT1g(:));
imT1g = imT1g / max3;
imT1g(imT1g<thr) = 0;

max4 = max(imFL(:));
imFL = imFL / max4;
imFL(imFL<thr) = 0;

figure;
subplot(1,4,1);imshow(imT1(:,:,80));title('T1');
subplot(1,4,2);imshow(imT2(:,:,80));title('T2');
subplot(1,4,3);imshow(imT1g(:,:,80));title('T1g');
subplot(1,4,4);imshow(imFL(:,:,80));title('FLAIR');


%%
slice = imT2(:,:,80);
slice = slice / max(slice(:));
slice(slice<exp(-4)) = 0;
thrs2 = multithresh(slice,2);
qim = imquantize(slice,thrs2);
qim(qim==2) = 1;
qim(qim~=3) = 0;
qim = bwareaopen(qim,20);

BW = bwperim(qim);
figure;imshow(BW);
%%
slice = adapthisteq(slice);
figure;imhist(slice);
%%
% figure;imshow(slice);
% slice = imadjust(slice);
level = graythresh(slice);
BW = im2bw(slice,level);
figure;imshow(BW);
C = ~BW;
D = -bwdist(C);
L = watershed(D);
slice(L==0) = 0;

B=bwdist(~imT1g);
C=-B;
ws = watershed(C);

imT1g(ws==0) = 0;
ws = double(ws);
ws = ws / max(ws(:));
%%
% Quantization for single example:
[X, Y, Z] = size(imT2);
threshT2 = multithresh(imT2, 2);

quantImage = zeros(X, Y, Z);
for z=1:Z
    C = mat2cell(imT2(:,:,z),[54,54,54,54],[80,80]);
    res = cell(size(C));
    for i=1:4
        for j=1:2
            thresh = multithresh(C{i,j}, 2);
            res{i,j} = imquantize(C{i,j},thresh);
        end
    end
    quantImage(:,:,z) = cell2mat(res);
end

threshFL = multithresh(imFL, 2);
quantImageFL = zeros(X, Y, Z);
for z=1:Z
    quantImageFL(:,:,z) = imquantize(imFL(:,:,z),threshFL);
end

%
edgeT1g = zeros(X, Y, Z);
for z=1:Z
    edgeT1g(:,:,z) = edge (imT1g(:,:,z),'Canny');
end

% Ceate predict mask
quantImage(quantImage~=3) = 0;
quantImageFL(quantImageFL~=3) = 0;


%
edgeT1g = edgeT1g & quantImageFL;
edgeT1g = cleanSegmentaionMask(edgeT1g,10);
[edgelist, edgeim, etype] = edgelink(edgeT1g(:,:,80), 2);

% Calculate predict mask and dicescore:
predict = quantImage & quantImageFL;
predictClean = cleanSegmentaionMask(predict,30);
predictClean = fillSegmentaionMask(predictClean);
diceScore = dice(predictClean,label);


%% quant in regions
zplane = 50;
slice = imT2(:,:,zplane);
C = mat2cell(slice,[54,54,54,54],[80,80]);
res = cell(size(C));
for i=1:4
    for j=1:2
        thresh = multithresh(C{i,j}, 2);
        res{i,j} = imquantize(C{i,j},thresh);
    end
end
resMat = cell2mat(res);

figure;subplot(1,3,1);imshow(resMat/3);
subplot(1,3,2);imshow(quantImage(:,:,zplane)/3);title('no patch')
subplot(1,3,3);imshow(label(:,:,zplane));imshow