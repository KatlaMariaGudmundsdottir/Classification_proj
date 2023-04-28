function predictedClasses = classifySingleGaussian(classes, features, columnMeans, covMatrices, testSet)
    predictedClasses = zeros(1, length(testSet));
    for k =  1:length(testSet)
        xk = testSet(k,:);
        pdf_k = zeros(1,classes);
        for C = 1:classes 
            pdf_k(C) = mvnpdf(xk,columnMeans(C,:), covMatrices(1+features*(C-1):features*C,:));
        end
        [~, predictedClasses(k)] = max(pdf_k);
    end
end