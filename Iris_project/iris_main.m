clear all

x1all = load('class_1','-ascii');
x2all = load('class_2', '-ascii');
x3all = load('class_3', '-ascii');

traininSet = 1:30;
testSet = 31:50;
features = ["l.sepals", "w.sepals", "l.petals", "w.petals"];

x1 = x1all;
x2 = x2all;   
x3 = x3all;

% x1 = [x1all(:,1) x1all(:,3) x1all(:,4)];
% x2 = [x2all(:,1) x2all(:,3) x2all(:,4)];
% x3 = [x3all(:,1) x3all(:,3) x3all(:,4)];

% x1 = [x1all(:,3) x1all(:,4)];
% x2 = [x2all(:,3) x2all(:,4)];
% x3 = [x3all(:,3) x3all(:,4)];
    
% x1 = [x1all(:,4)];
% x2 = [x2all(:,4)];
% x3 = [x3all(:,4)];

x1training = [x1(traininSet,:)];
x1test = [x1(testSet,:)];
tk1training = kron([1;0;0],ones(size(traininSet)));
tk1test = kron([1;0;0],ones(size(testSet)));
x2training = [x2(traininSet,:)];
x2test = [x2(testSet,:)];
tk2training = kron([0;1;0],ones(size(traininSet)));
tk2test = kron([0;1;0],ones(size(testSet)));
x3training = [x3(traininSet,:)];
x3test = [x3(testSet,:)];
tk3training = kron([0;0;1],ones(size(traininSet)));
tk3test = kron([0;0;1],ones(size(testSet)));
xtraining = [x1training; x2training; x3training];
xtest = [x1test; x2test; x3test];
tktraining = [tk1training,tk2training,tk3training];
tktest = [tk1test, tk2test, tk3test];

C = 3;

[Ntot,dimx] = size(xtraining);
 
alpha = 0.005;
iterations = 10000;

[W,MSE] = trainingLinearClassifier(xtraining,tktraining, alpha, iterations); 
[g_training,confmat_training] = linearClassifier(xtraining, tktraining, W);
[g_test,confmat_test] = linearClassifier(xtest, tktest, W);


errorRateTrainingSet = calculateErrorRate(confmat_training, length(traininSet));
training_fig = plotConfusionMatrix(confmat_training, 'Confusion Matrix Training Set', errorRateTrainingSet);

errorRateTestSet = calculateErrorRate(confmat_test, length(testSet));
test_fig = plotConfusionMatrix(confmat_test, 'Confusion Matrix Test Set',errorRateTestSet);

histogram = plotHistograms(x1all, x2all, x3all, features);
