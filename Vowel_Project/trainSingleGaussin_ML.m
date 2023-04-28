function [columnMeans, covMatrices, diagCovMatrices] = trainSingleGaussian_ML(classes, features, trainingPerClass, trainSet)
    % Initialize outputs
    columnMeans = zeros(classes,features);
    covMatrices = zeros(classes*features,features);
    diagCovMatrices = zeros(classes*features,features);

    for i = 1:classes
        index1 = 1+trainingPerClass*(i-1);
        index2 = trainingPerClass*(i);
        classTrainingSet = trainSet(index1:index2,:);
        columnMeans(i,:) = mean(classTrainingSet,1);
        CovMatrix = cov(classTrainingSet);
        DiagCovMatrix = diag(diag(CovMatrix));
        covMatrices(1+features*(i-1):features*i,:) = CovMatrix;
        diagCovMatrices(1+features*(i-1):features*i,:) = DiagCovMatrix;
    end
end