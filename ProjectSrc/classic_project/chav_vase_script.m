%% Using quantization for BARTS HG tumor segmentation:

% 1. Load the data

clear all;
%addpath(genpath('/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc'))
% load the image matrix named Im
load('dataBN.mat','im')
% load the label matrix, named gt4
load('gt4.mat')

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
predictClean = cleanSegmentaionMask(predict,30);
predictClean = fillSegmentaionMask(predictClean);
diceScore = dice(predictClean,label);

imT2 = imT2 / max(imT2(:));

%% plot slice of phi
zSlice = 80;
[X, Y] = size(phi(:,:,zSlice));

edge = zeros(X,Y);
edge(phi(:,:,zSlice)==0) = 1;
surf(phi(:,:,zSlice));
colormap(jet);
hold on; contour3(edge,'k');

%% Chane Vase parametres
smooth_weight = 1; 
image_weight = 1e-6; 
delta_t = 4; 

%% Load data
data = load_all_data()
label = load_all_labels()

%% Loop over some examples

numOfExamples = 5;
diceScoreArray = zeros(numOfExamples, 1);
sensitivityArray = zeros(numOfExamples, 1);
specificityArray = zeros(numOfExamples, 1);
CVpredictCell = {}; % cell array for the CV predict masks

startTime = tic;
for i=1:numOfExamples
    
    [diceScoreArray(i), predict] = quantizationT2andFLSegmentation(data(i).f,label(i).f);
    predictClean = cleanSegmentaionMask(predict,30);
    predictClean = fillSegmentaionMask(predictClean);
    
    labels = double(label(i).f);
    labels(labels~=0) = 1;
    orgImg = data(i).f;
    orgMod = orgImg(:,:,:,2); % extract T2 mod
    
    % Chan Vase methode
    phi = ac_reinit(predictClean-.5); 

    for j = 1:10
        phi = ac_ChanVese_model(orgMod, phi, smooth_weight, image_weight, delta_t, 1); 
    end
    
    % from phi to bineary-mask
    CVpredict = zeros(size(phi));
    CVpredict(phi>0) = 1;
    CVpredictCell{i} = CVpredict;
    % measure parameters
    diceScoreArray(i) = dice(predictClean,labels);
    sensitivityArray(i) = sensitivity(labels, predictClean);
    specificityArray(i) = specificity(labels, predictClean);
end

avarageDiceScore = sum(diceScoreArray) / numOfExamples; 
avarageSeneScore = sum(sensitivityArray) / numOfExamples; 
avarageSpeceScore = sum(specificityArray) / numOfExamples; 

endTime = toc;
difTime = startTime - endTime;
%% Single example run:
phi = ac_reinit(predictClean-.5); 

for i = 1:50
    phi = ac_ChanVese_model(imT2, phi, smooth_weight, image_weight, delta_t, 1); 
    if mod(i,10) == 0
        figure; imshow(phi(:,:,80));
    end
end
