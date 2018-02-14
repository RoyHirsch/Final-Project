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

clear all;

% 1. Load the data - 4 images
%  X = training data of numExp

numExp = 4;

addpath(genpath('/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc'))
data = load_all_data()
label = load_all_labels()

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

Y(Y~=0) = 1;

%% 3. Reduce for train and validation data:

% parameters:
params.train = 100000;
params.train = 50000;
params.val = 1000;

ind = randi([100000 20000000],1,params.train);
indVal = randi([100000 20000000],1,params.val);
Xtrain = X(ind,:);
Ytrain = Y(ind);
Xval = X(indVal,:);
Yval = Y(indVal);

%% 3. Train simple SVM model
SVMModel = fitcsvm(Xtrain,Ytrain,'RemoveDuplicates','on');

%% 4. Validation Accuracy:
[Ypredict,score] = predict(SVMModel,Xval);
accuracy = sum(Ypredict==Yval)/length(Yval);

%% 5. Predict for the new test data
% Load new data:
Xtest = reshape(data(5).f,[],4);
Ytest = reshape(label(5).f,[],1);
%
[label,score] = predict(SVMModel,Xtest);
Ypredict =  reshape(label,[H,W,D]);

%% 6. dice score
dice = dice(Ypredict,Ytest);

%% 7. Interactive test:
figure;imshow3D(double(gt4)/4);
[H, W, D, C] = size(data(6).f);

Xtest = reshape(data(6).f,[],4);
Ytest = reshape(label(6).f,H,W,D);
%
[labelP,score] = predict(SVMModel,Xtest);
Ypredict =  reshape(labelP,[H,W,D]);

%% 6. dice score
dice = dice(Ypredict,Ytest)

%% 7. Interactive test:
figure;imshow3D(Ytest/4);
figure;imshow3D(Ypredict);