function [trainSet, testSet, trainLabels, testLabels] = seperateData(classes, features, data, testPerClass, trainingPerClass, dataPerClass)
    numMen = 45; trainNumMen = 22; testNumMen = 23;
    numWomen = 48; trainNumWomen = 24; testNumWomen = 24;
    numBoys = 27; trainNumBoys = 14; testNumBoys = 13;
    numGirls = 19; trainNumGirls = 10; testNumGirls = 9;

    % Initialize Sets
    testSet = zeros(testPerClass*classes,features);
    testLabels = zeros(1,testPerClass*classes);
    trainSet = zeros(trainingPerClass*classes,features);
    trainLabels = zeros(1,trainingPerClass*classes);
    noSplitTrainSet = zeros(trainingPerClass*classes,features); % training set with no splitting of data for comparison
    noSplitTestSet = zeros(testPerClass*classes,features);
    
    
    % Seperating data
    % Restructuring data into a 3-dimensional array
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
        noSplitTrainSet(1+trainingPerClass*(i-1):trainingPerClass*i, :) = data(1:trainingPerClass, i, :);
        trainLabels(1+trainingPerClass*(i-1):trainingPerClass*i) = i.*ones(trainingPerClass, 1);
        
        % Testing data
        menTestData = reshape(data(testMenIndex1:testMenIndex2, i, :), [testNumMen, features]);
        womenTestData = reshape(data(testWomenIndex1:testWomenIndex2, i, :), [testNumWomen, features]);
        boysTestData = reshape(data(testBoysIndex1:testBoysIndex2, i, :), [testNumBoys, features]);
        girlsTestData = reshape(data(testGirlsIndex1:testGirlsIndex2, i, :), [testNumGirls, features]);
        testSet(1+testPerClass*(i-1):testPerClass*i, :) = [menTestData; womenTestData; boysTestData; girlsTestData];
        noSplittTestSet(1+testPerClass*(i-1):testPerClass*i, :) = data(trainingPerClass+1:139, i, :);
        testLabels(1+testPerClass*(i-1):testPerClass*i) = i.*ones(testPerClass, 1);
    end 
end