%% Import data from text file.
clear all
% Initialize variables.
currentFolder = pwd;
filename = fullfile(currentFolder, 'vowdata_nohead.dat');

% Format for each line of text:
formatSpec = '%5s%4f%4f%5f%5f%5f%5f%5f%5f%5f%5f%5f%5f%5f%5f%5f%[^\n\r]';

% Open the text file.
fileID = fopen(filename,'r');

% Read columns of data according to the format.
% This call is based on the structure of the file used to generate this code. If an error occurs for a different file, try regenerating the code from the Import Tool.
dataArray = textscan(fileID, formatSpec, 'Delimiter', '', 'WhiteSpace', '', 'TextType', 'string',  'ReturnOnError', false);

% Remove white space around all cell columns.
dataArray{1} = strtrim(dataArray{1});

% Close the text file.
fclose(fileID);

% Create output variable
vowdatanohead = table(dataArray{1:end-1}, 'VariableNames', {'identifier','durationMS','f0ss','F1_ss','F2_ss','F3_ss','F4_ss','F1_20','F2_20','F3_20','F1_50','F2_50','F3_50','F1_80','F2_80','F3_80'});

% Clear temporary variables
clearvars filename formatSpec fileID dataArray ans;

%% Classification variables
classes = 12;
features = 15;
dataPerClass = 139;
trainingPerClass = 70;
testPerClass = dataPerClass-trainingPerClass;

numMen = 45; trainNumMen = 22; testNumMen = 23;
numWomen = 48; trainNumWomen = 24; testNumWomen = 24;
numBoys = 27; trainNumBoys = 14; testNumBoys = 13;
numGirls = 19; trainNumGirls = 10; testNumGirls = 9;

testSet = zeros(testPerClass*classes,features);
testLabels = zeros(1,testPerClass*classes);
trainSet = zeros(trainingPerClass*classes,features);
trainLabels = zeros(1,trainingPerClass*classes);
noSplitTrainSet = zeros(trainingPerClass*classes,features); % training set with no splitting of data for comparison
noSplitTestSet = zeros(testPerClass*classes,features);

%% Seperating training and test data
% Restructuring data into a 4-dimensional array
data = table2array(vowdatanohead(:, 2:16));
data = reshape(data, [dataPerClass, classes, features]);

% Indexes for splitting data
trainWomenIndex1 = numMen+1;
trainBoysIndex1 = trainWomenIndex1 + numWomen;
trainGirlsIndex1 = trainBoysIndex1 + numBoys;

trainMenIndex2 = trainNumMen;
trainWomenIndex2 = trainWomenIndex1 + trainNumWomen-1;
trainBoysIndex2 = trainBoysIndex1 + trainNumBoys-1;
trainGirlsIndex2 = trainGirlsIndex1 + trainNumGirls-1;

testMenIndex1 = trainMenIndex2+1;
testWomenIndex1 = trainWomenIndex2+1; 
testBoysIndex1 = trainBoysIndex2 +1;
testGirlsIndex1 = trainGirlsIndex2 + 1;

testMenIndex2 = numMen;
testWomenIndex2 = testWomenIndex1 + testNumWomen-1;
testBoysIndex2 = testBoysIndex1 + testNumBoys-1;
testGirlsIndex2 = testGirlsIndex1 + testNumGirls-1;

for i = 1:classes
    % Training data
    menTrainData = reshape(data(1:trainMenIndex2, i, :), [trainNumMen, features]);
    womenTrainData = reshape(data(trainWomenIndex1:trainWomenIndex2, i, :), [trainNumWomen, features]);
    boysTrainData = reshape(data(trainBoysIndex1:trainBoysIndex2, i, :), [trainNumBoys, features]);
    girlsTrainData = reshape(data(trainGirlsIndex1:trainGirlsIndex2, i, :), [trainNumGirls, features]);
    trainSet(1+trainingPerClass*(i-1):trainingPerClass*i, :) = [menTrainData; womenTrainData; boysTrainData; girlsTrainData];
    trainLabels(1+trainingPerClass*(i-1):trainingPerClass*i) = i.*ones(trainingPerClass, 1);
    
    % Testing data
    menTestData = reshape(data(testMenIndex1:testMenIndex2, i, :), [testNumMen, features]);
    womenTestData = reshape(data(testWomenIndex1:testWomenIndex2, i, :), [testNumWomen, features]);
    boysTestData = reshape(data(testBoysIndex1:testBoysIndex2, i, :), [testNumBoys, features]);
    girlsTestData = reshape(data(testGirlsIndex1:testGirlsIndex2, i, :), [testNumGirls, features]);
    testSet(1+testPerClass*(i-1):testPerClass*i, :) = [menTestData; womenTestData; boysTestData; girlsTestData];
    testLabels(1+testPerClass*(i-1):testPerClass*i) = i.*ones(testPerClass, 1);
end 

for i = 1:classes
    noSplitTrainSet(1+trainingPerClass*(i-1):trainingPerClass*i, :) = data(1:trainingPerClass, i, :);
    noSplittTestSet(1+testPerClass*(i-1):testPerClass*i, :) = data(trainingPerClass+1:139, i, :);
end


%% Training
covdef = zeros(12,1);
columnMeans = zeros(classes,features);
covMatrcies = zeros(classes*features,features);
diagCovMatrices = zeros(classes*features,features);
for i = 1:classes
    index1 = 1+trainingPerClass*(i-1);
    index2 = trainingPerClass*(i);
    res = trainSet(index1:index2,:);
    columnMeans(i,:) = mean(res,1);
    tempCovMatrix = cov(res);
    tempDiagCovMatrix = diag(diag(tempCovMatrix));
    covMatrcies(1+features*(i-1):features*i,:) = tempCovMatrix;
    diagCovMatrixes(1+features*(i-1):features*i,:) = tempDiagCovMatrix;
end

%% Classification
predictedClasses = zeros(1, length(testSet));
for k =  1:length(testSet)
    xk = testSet(k,:);
    pdf_k = zeros(1,classes);
    for C = 1:classes 
        pdf_k(C) = mvnpdf(xk,columnMeans(C,:), covMatrcies(1+features*(C-1):features*C,:));
    end
    [~, predictedClasses(k)] = max(pdf_k);
end

confmat = confusionmat(testLabels, predictedClasses);
errorRate = calculateErrorRate(confmat,testPerClass);
plotConfusionMatrixGPT(confmat, 'Confusion Matrix for Test Set w. Full Covariance', errorRate)


