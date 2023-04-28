function predictedClasses = classifyGMM(classes, testSet, GMModels)
    predictedClasses = zeros(1, length(testSet));
    for k =  1:length(testSet)
        xk = testSet(k,:);
        pdf_k = zeros(1,classes);
        for C = 1:classes 
            for i = 1:GMModels{1}.NumComponents
                pdf_k(C) = pdf_k(C) + GMModels{C}.ComponentProportion(i)*mvnpdf(xk, GMModels{C}.mu(i,:), GMModels{C}.Sigma(:,:,i));
            end
        end
        [~, predictedClasses(k)] = max(pdf_k);
    end
end