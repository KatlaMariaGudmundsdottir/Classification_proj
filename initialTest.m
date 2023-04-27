clear all

x1all = load('class_1','-ascii');
x2all = load('class_2', '-ascii');
x3all = load('class_3', '-ascii');

traininSet = 21:50;
testSet = 1:20;



x1training = [x1all(traininSet,:)];
x1test = [x1all(testSet,:)];
tk1training = kron([1;0;0],ones(size(traininSet)));
tk1test = kron([1;0;0],ones(size(testSet)));
x2training = [x2all(traininSet,:)];
x2test = [x2all(testSet,:)];
tk2training = kron([0;1;0],ones(size(traininSet)));
tk2test = kron([0;1;0],ones(size(testSet)));
x3training = [x3all(traininSet,:)];
x3test = [x3all(testSet,:)];
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
training_fig = plotConfusionMatrixGPT(confmat_training, 'Confusion Matrix Test Set', errorRateTrainingSet);
filename_training = 'confmat_test_2.png';
exportgraphics(training_fig,filename_training) 


errorRateTestSet = calculateErrorRate(confmat_test, length(testSet));
test_fig = plotConfusionMatrix(confmat_test, 'Confusion Matrix Training Set');
filename_test = 'confmat_training_2.png';
exportgraphics(test_fig,filename_test) 


