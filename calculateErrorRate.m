function errorRate = calculateErrorRate(confusionMatrix,testPerClass)
    C = length(confusionMatrix);
    incorrectPred = 0;
    for i = 1:C
        incorrectPred = incorrectPred + (testPerClass - confusionMatrix(i, i));
    end
    errorRate = incorrectPred/(testPerClass*C);
end

