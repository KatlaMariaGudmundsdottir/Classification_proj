%% Import data from text file.
clear all
currentFolder = pwd;
filename = fullfile(currentFolder, 'vowdata_nohead.dat');
formatSpec = '%5s%4f%4f%5f%5f%5f%5f%5f%5f%5f%5f%5f%5f%5f%5f%5f%[^\n\r]'; % Format for each line of text:
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', '', 'WhiteSpace', '', 'TextType', 'string',  'ReturnOnError', false);% Read columns of data according to the format.
dataArray{1} = strtrim(dataArray{1}); % Remove white space around all cell columns.
fclose(fileID);
vowdatanohead = table(dataArray{1:end-1}, 'VariableNames', {'identifier','durationMS','f0ss','F1_ss','F2_ss','F3_ss','F4_ss','F1_20','F2_20','F3_20','F1_50','F2_50','F3_50','F1_80','F2_80','F3_80'});
clearvars filename formatSpec fileID dataArray ans; % Clear temporary variables
data = table2array(vowdatanohead(:, 2:16));


%% Classification variables
classes = 12;
features = 15;
dataPerClass = 139;
trainingPerClass = 70;
testPerClass = dataPerClass-trainingPerClass;


%% Seperating data
[trainSet, testSet, trainLabels, testLabels] = seperateData(classes, features, data, testPerClass, trainingPerClass, dataPerClass);

%% Training
% Training single gaussian model with ML training
[columnMeans, covMatrices, diagCovMatrices] = trainSingleGaussin_ML(classes, features, trainingPerClass, trainSet);

% Training Gaussian Mixture Model with ME training
numMixtures = 2;
GMModels = trainGMM_EM(classes, trainingPerClass, trainSet, numMixtures);


%% Classification
singleGaussian_predictedClasses = classifySingleGaussian(classes, features, columnMeans, covMatrices, testSet);
GMM_predictedClasses = classifyGMM(classes, testSet, GMModels);


%% Plot Confusion matric and error rate
% Single Gaussian
confmat = confusionmat(testLabels, singleGaussian_predictedClasses);
errorRate = calculateErrorRate(confmat,testPerClass);
plotConfusionMatrix(confmat, 'Confusion Matrix for Test Set w. Full Covariance', errorRate)

% GMM
confmatGMM = confusionmat(testLabels, GMM_predictedClasses);
errorRate = calculateErrorRate(confmatGMM,testPerClass);
plotConfusionMatrix(confmatGMM, 'Confusion Matrix for GMM ', errorRate)


