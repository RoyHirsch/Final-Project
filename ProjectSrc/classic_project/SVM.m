% SVM model for brain tumor classification on BARTS data
%
% Simple tryout to train SVM model over BARTS dataset in matlab.
%
% SVM: creating an optimazid hyperplane to separate between two classes [1,-1]
% the optimal hyperplane have a maximum margin, the hyperplane equation:
% f(x) =x'beta + b
%   x - the training data
%   beta - the hyperplane normal (hyperplane parameters)
%   b - bias
% 
% See more info:
% https://www.mathworks.com/help/stats/support-vector-machines-for-binary-classification.html#bs3tbev-16
% https://www.mathworks.com/help/stats/fitcsvm.html#bt8v_23-1

% ideas:
% soft edges for nonseparable data
% adding OutlierFraction

clear all;

% 1. Load the data
addpath(genpath('/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc'))
% load the image matrix named Im
load('/Data/BRATS_HG0001/dataBN.mat','im')
% load the label matrix, named gt4
load('/Data/BRATS_HG0001/gt4.mat')

% 2. Pre-process and re-shape the data
X = double(im);
[H, W, D, C] = size(im);
X = reshape(X,[H*W*D,C]);

Y = double(gt4);
Y = reshape(Y,[H*W*D,1]);
Y(Y~=0) = 1;

%% 3. Reduce for train and validation data:

% parameters:
<<<<<<< HEAD
params.train = 10000;
=======
params.train = 50000;
>>>>>>> 4fdd9ba27907b8048cc1664a3644950ecfaca6f0
params.val = 1000;

ind = randi([1 5000000],1,params.train);
indVal = randi([1000000 6000000],1,params.val);
Xtrain = X(ind,:);
Ytrain = Y(ind);
Xval = X(indVal,:);
Yval = Y(indVal);

%% 3. Train simple SVM model
SVMModel = fitcsvm(Xtrain,Ytrain,'RemoveDuplicates','on');

%% 4. Validation Accuracy:
[Ypredict,score] = predict(SVMModel,Xval);
accuracy = sum(Ypredict==Yval)/length(Yval);

%% 5. Predict for the whole data:
[label,score] = predict(SVMModel,X);
Ypredict =  reshape(label,[H,W,D]);

<<<<<<< HEAD
=======
%% 5.5 Predict for a different image
load('/Data/BRATS_HG0004/dataBN.mat','im')
% load the label matrix, named gt4
load('/Data/BRATS_HG0004/gt4.mat')
Xpredict = double(im);
Xpredict = reshape(Xpredict,[],C);

[label,score] = predict(SVMModel,Xpredict);
Ypredict =  reshape(label,H,W,[]);

>>>>>>> 4fdd9ba27907b8048cc1664a3644950ecfaca6f0
%% 6. dice score
dice = dice(Ypredict,double(gt4));

%% 7. Interactive test:
figure;imshow3D(double(gt4)/4);
figure;imshow3D(Ypredict);