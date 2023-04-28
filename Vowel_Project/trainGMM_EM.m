function GMModels = trainGMM_EM(classes, trainingPerClass, trainSet, numMixtures)
    GMModels = cell(1,classes); % initialize cell array
    for i = 1:classes
        index1 = 1 + trainingPerClass*(i-1);
        index2 = trainingPerClass*i;
        classTrainingSet = trainSet(index1:index2,:);
        options = statset('MaxIter',1000, 'TolFun', 1e-10);
        GMModels{i} = fitgmdist(classTrainingSet, numMixtures, 'RegularizationValue', 1e-12,'CovarianceType','diagonal','Options',options); % store GMM object in cell array
    end
end