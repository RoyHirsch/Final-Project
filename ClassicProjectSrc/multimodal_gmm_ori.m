%%  ---------- Train gmm for multi-modal input data: -------------------

clear all;
%% load data
addpath(genpath('\Users\אורי\Documents\GitHub\Final-Project\ProjectSrc'))
% load the image matrix named Im
load('/Data/BRATS_HG0004/dataBN.mat','im')
% load the label matrix, named gt4
load('/Data/BRATS_HG0004/gt4.mat')

im = double(im);
label = double(gt4);
%% Pre-process the data:
% Training data in [n,d] struct where:
% n - number of voxals
% d - number of modulities.
% 
% The model will calculate the probability of the i's voxal
% (1X4 double vector) to be a part of the 4D geussian distribiution.
% 
% 
[H, W, D, C] = size(im);
dataMatrix = zeros(H*W*D,3); % dataMatrix - a reshape of input data to shape(H*W*D,c) struct
for i=1:C
numChannels = 2;
[H, W, D, C] = size(im);
dataMatrix = zeros(H*W*D,numChannels); % dataMatrix - a reshape of input data to shape(H*W*D,c) struct
end
for i=1:numChannels
    modeImg = im(:,:,:,i*2);
    minMode = min(modeImg(:));
    maxMode = max(modeImg(:));
    dataMatrix(:,i) = modeImg(:) / maxMode;
%     dataMatrix(:,i) = modeImg(modeImg>minMode);
end

%% Train GMM:
%% Data manipulation
dataMatrix=image_adjustment(dataMatrix,0,0,1,2);
%% initialization
numIter = 400;
thresh=10^-5;
Mu=[0.1,0.1;0.1,0.5;0.5,0.1;0.8,0.8];
Sigma(:,:,1)=[0.025,0.01;0.01,0.025];
Sigma(:,:,2)=[0.025,0.01;0.01,0.025];
Sigma(:,:,3)=[0.025,0.01;0.01,0.025];
Sigma(:,:,4)=[0.125,0.01;0.01,0.125];
PComponents = [0.25,0.25,0.25,0.25];
S = struct('mu',Mu,'Sigma',Sigma,'ComponentProportion',PComponents);
%%
gm = fitgmdist(dataMatrix,4,'RegularizationValue',0.001,'Start',S,'Options',statset('MaxIter',numIter));
%gm = fitgmdist(dataMatrix,4,'RegularizationValue',0.001,'Start',S,'Options',statset('MaxIter',numIter));

%% Create PDF matrix, dimentions[n,classes].
% In each (i,j) cell: the distribiuotion the the i'th voxal to be a part of
% the j's class.
% 
% The function 'mvnpdf' (mvnpdf(X,mu,sigma)) creats a multimodat guessian distribioution with:
% X - [n,d], d - the modalities
% mu = [classes,d] vector of means per modality (dimention)
% sigma - [d,d] covarience matrix for the class
% 

meanV = gm.mu;
sigmaV = gm.Sigma;
propV = gm.ComponentProportion;
classes = length(meanV(:,1));

% Sort by proportion (the smallest is the tumor class). 
% For getting a consistent classes for every run

% [val,ind] = intersect(propV,sort(propV));
% meanV = meanV(ind,:);
% sigmaV = sigmaV(:,:,ind);
% propV = propV(ind);

pdfM = zeros(H*W*D,classes); 
sumM = zeros(H*W*D,1);
for i=1:classes
    pdfM(:,i) = mvnpdf(dataMatrix,meanV(i,:),sigmaV(:,:,i));
    sumM = sumM + pdfM(:,i)*PComponents(i);
    %sumM = sumM + pdfM(:,i)*propV(i);
end

for i=1:classes
     pdfM(:,i) =  pdfM(:,i)*PComponents(i) ./ sumM;
     %pdfM(:,i) =  pdfM(:,i)*propV(i) ./ sumM;
end
%%
images = reshape(dataMatrix,H,W,D,2);
figure;
imshow(images(:,:,100,1));
figure;
imshow(images(:,:,100,2));
figure;
imhist(images(:,:,100,1),256);

%% Reshape pdfM to 3D probability matrices

% Each classes dim is a 3D probability matrix for this class.
pdfM = reshape(pdfM,H,W,D,classes);
for i=1:classes
        figure;
        imagesc(pdfM(:,:,80,i)); colorbar;title(['PDF matrix of geussian num ',num2str(i),': '])
end

%% Create segmentation matrix
% By max pooling for all classes in each voxal (i,j).
segMetrix = zeros(H,W,D);

for i=1:H
    for j=1:W
        for k=1:D
            [x,IND]=max(pdfM(i,j,k,:));
             segMetrix(i,j,k)=IND;
        end
    end
end
figure; imagesc(segMetrix(:,:,80)); colorbar;

%% Evaluate the segmentation:
predict = zeros(size(segMetrix));
predict(segMetrix==4) = 1;
diceScore = dice(predict,label)
