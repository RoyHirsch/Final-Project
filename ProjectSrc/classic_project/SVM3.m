% SVM model for brain tumor classification on BARTS data
%
% Two modalities only: T2 and FLAIR
% Trained over 4 train examples
% Unique data input
% 

clear all;

% 1. Load the data - train over numExp examples:

numExp = 4;

addpath(genpath('/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc'))
data = load_all_data()
label = load_all_labels()

[H, W, D, C] = size(data(1).f);

X = zeros(1,4);
for i=1:numExp
    temp = reshape(data(i).f,[],4);
    X = vertcat(X,temp);
end

Y = zeros(1,1);
for i=1:numExp
    temp = reshape(label(i).f,[],1);
    Y = vertcat(Y,temp);
end

% reduce dimention of X
X = X(:,[2,4]);
Y(Y~=0) = 1;

%% Train over one example:

% load the image matrix named Im
load('/Data/BRATS_HG0001/dataBN.mat','im')
% load the label matrix, named gt4
load('/Data/BRATS_HG0001/gt4.mat')

% 2. Pre-process and re-shape the data
X = double(im);
X = image_adjustment(X,0,0,1,2);

[H, W, D, C] = size(im);
X = reshape(X,[H*W*D,C]);

Y = double(gt4);
Y = reshape(Y,[H*W*D,1]);
X = X(:,[2,4]);
Y(Y~=0) = 1;
%% 3. Reduce for train and validation data:

% parameters:
params.train = 400000;
params.val = 1000;
params.countData = 6000000;

ind = randi([1 params.countData],1,params.train);
Xtrain = X(ind,:);
Ytrain = Y(ind);

% Extract unique training data:
% use temp data struct for unique values only
tempMatrix = zeros(params.train,3);
tempMatrix(:,1:2) = Xtrain;
tempMatrix(:,3) = Ytrain;
tempMatrixUnique = unique(tempMatrix,'rows');

Xtrain = tempMatrixUnique(:,1:2);
Ytrain =  tempMatrixUnique(:,3);

% Extract validation data:
indVal = randi([1 params.countData],1,params.val);
Xval = X(indVal,:);
Yval = Y(indVal,:);
%% 3. Train simple SVM model
SVMModel = fitcsvm(Xtrain,Ytrain);

%% 4. Validation Accuracy:
[Ypredict,score] = predict(SVMModel,Xval);
accuracy = sum(Ypredict==Yval)/length(Yval);

%% 5. Predict for a different image
load('/Data/BRATS_HG0003/dataBN.mat','im')
% load the label matrix, named gt4
load('/Data/BRATS_HG0003/gt4.mat')

Xtest = double(im);
Xtest = reshape(Xtest,[],4);
Xtest = Xtest(:,[2,4]);
Ytest = double(gt4);
Ytest(Ytest~=0) = 1;

[label,score] = predict(SVMModel,Xtest);
Ypredict =  reshape(label,H,W,[]);

%% 6. dice score
dice = dice(Ypredict,Ytest);

%% 7. Interactive test:
figure;imshow3D(Ytest);
figure;imshow3D(Ypredict);