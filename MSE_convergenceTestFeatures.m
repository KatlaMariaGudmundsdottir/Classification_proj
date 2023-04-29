clear all

x1all = load('class_1','-ascii');
x2all = load('class_2', '-ascii');
x3all = load('class_3', '-ascii');

traininSet = 1:30;
testSet = 31:50;
features = ["l.sepals", "w.sepals", "l.petals", "w.petals"];

x1_all = x1all;
x2_all = x2all;   
x3_all = x3all;

x1_3features = [x1all(:,1) x1all(:,3) x1all(:,4)];
x2_3features = [x2all(:,1) x2all(:,3) x2all(:,4)];
x3_3features = [x3all(:,1) x3all(:,3) x3all(:,4)];

x1_2features = [x1all(:,3) x1all(:,4)];
x2_2features = [x2all(:,3) x2all(:,4)];
x3_2features = [x3all(:,3) x3all(:,4)];
    
x1_1feature = [x1all(:,4)];
x2_1feature = [x2all(:,4)];
x3_1feature = [x3all(:,4)];

x1training_all = [x1_all(traininSet,:)];
x1test_all = [x1_all(testSet,:)];
tk1training_all = kron([1;0;0],ones(size(traininSet)));
tk1test_all = kron([1;0;0],ones(size(testSet)));
x2training_all = [x2_all(traininSet,:)];
x2test_all = [x2_all(testSet,:)];
tk2training_all = kron([0;1;0],ones(size(traininSet)));
tk2test_all = kron([0;1;0],ones(size(testSet)));
x3training_all = [x3_all(traininSet,:)];
x3test_all = [x3_all(testSet,:)];
tk3training_all = kron([0;0;1],ones(size(traininSet)));
tk3test_all = kron([0;0;1],ones(size(testSet)));
xtraining_all = [x1training_all; x2training_all; x3training_all];
xtest_all = [x1test_all; x2test_all; x3test_all];
tktraining_all = [tk1training_all,tk2training_all,tk3training_all];
tktest_all = [tk1test_all, tk2test_all, tk3test_all];

x1training_3features = [x1_3features(traininSet,:)];
x1test_3features = [x1_3features(testSet,:)];
tk1training_3features = kron([1;0;0],ones(size(traininSet)));
tk1test_3features = kron([1;0;0],ones(size(testSet)));
x2training_3features = [x2_3features(traininSet,:)];
x2test_3features = [x2_3features(testSet,:)];
tk2training_3features = kron([0;1;0],ones(size(traininSet)));
tk2test_3features = kron([0;1;0],ones(size(testSet)));
x3training_3features = [x3_3features(traininSet,:)];
x3test_3features = [x3_3features(testSet,:)];
tk3training_3features = kron([0;0;1],ones(size(traininSet)));
tk3test_3features = kron([0;0;1],ones(size(testSet)));
xtraining_3features = [x1training_3features; x2training_3features; x3training_3features];
xtest_3features = [x1test_3features; x2test_3features; x3test_3features];
tktraining_3features = [tk1training_3features,tk2training_3features,tk3training_3features];
tktest_3features = [tk1test_3features, tk2test_3features, tk3test_3features];

x1training_2features = [x1_2features(traininSet,:)];
x1test_2features = [x1_2features(testSet,:)];
tk1training_2features = kron([1;0;0],ones(size(traininSet)));
tk1test_2features = kron([1;0;0],ones(size(testSet)));
x2training_2features = [x2_2features(traininSet,:)];
x2test_2features = [x2_2features(testSet,:)];
tk2training_2features = kron([0;1;0],ones(size(traininSet)));
tk2test_2features = kron([0;1;0],ones(size(testSet)));
x3training_2features = [x3_2features(traininSet,:)];
x3test_2features = [x3_2features(testSet,:)];
tk3training_2features = kron([0;0;1],ones(size(traininSet)));
tk3test_2features = kron([0;0;1],ones(size(testSet)));
xtraining_2features = [x1training_2features; x2training_2features; x3training_2features];
xtest_2features = [x1test_2features; x2test_2features; x3test_2features];
tktraining_2features = [tk1training_2features,tk2training_2features,tk3training_2features];
tktest_2features = [tk1test_2features, tk2test_2features, tk3test_2features];

x1training_1feature = [x1_1feature(traininSet,:)];
x1test_1feature = [x1_1feature(testSet,:)];
tk1training_1feature = kron([1;0;0],ones(size(traininSet)));
tk1test_1feature = kron([1;0;0],ones(size(testSet)));
x2training_1feature = [x2_1feature(traininSet,:)];
x2test_1feature = [x2_1feature(testSet,:)];
tk2training_1feature = kron([0;1;0],ones(size(traininSet)));
tk2test_1feature = kron([0;1;0],ones(size(testSet)));
x3training_1feature = [x3_1feature(traininSet,:)];
x3test_1feature = [x3_1feature(testSet,:)];
tk3training_1feature = kron([0;0;1],ones(size(traininSet)));
tk3test_1feature = kron([0;0;1],ones(size(testSet)));
xtraining_1feature = [x1training_1feature; x2training_1feature; x3training_1feature];
xtest_1feature = [x1test_1feature; x2test_1feature; x3test_1feature];
tktraining_1feature = [tk1training_1feature,tk2training_1feature,tk3training_1feature];
tktest_1feature = [tk1test_1feature, tk2test_1feature, tk3test_1feature];

C = 3;

% [Ntot,dimx] = size(xtraining);
 
alpha = 0.005;
iterations = 10000;
[W_all,MSE_all] = trainingLinearClassifier(xtraining_all,tktraining_all, alpha, iterations); 
[W_3features,MSE_3features] = trainingLinearClassifier(xtraining_3features,tktraining_3features, alpha, iterations);
[W_2features,MSE_2features] = trainingLinearClassifier(xtraining_2features,tktraining_2features, alpha, iterations);
[W_1feature,MSE_1feature] = trainingLinearClassifier(xtraining_1feature,tktraining_1feature, alpha, iterations);



figure('Units', 'inches', 'Position', [0, 0, 12, 6]);
plot(1:iterations, MSE_all, 'LineWidth', 3);
hold on;
plot(1:iterations, MSE_3features, 'LineWidth', 3);
plot(1:iterations, MSE_2features, 'LineWidth', 3);
plot(1:iterations, MSE_1feature, 'LineWidth', 3);
hold off;
title('MSE vs. Iterations for Different Feature Sets', 'FontSize', 16);
xlabel('Iterations', 'FontSize', 14);
ylabel('MSE', 'FontSize', 14);
legend('All Features', '3 Features', '2 Features', '1 Feature', 'FontSize', 12);
% Remove extra whitespace around the plot
set(gca,'LooseInset',max(get(gca,'TightInset'), 0.02))
% Save figure as a PNG file with minimized whitespace
filename = 'MSE_plot_fature_comp.png';
exportgraphics(gcf, filename, 'BackgroundColor', 'none');


% 
% [g_training,confmat_training] = linearClassifier(xtraining, tktraining, W);
% [g_test,confmat_test] = linearClassifier(xtest, tktest, W);
% 
% 
% errorRateTrainingSet = calculateErrorRate(confmat_training, length(traininSet));
% training_fig = plotConfusionMatrixGPT(confmat_training, 'Confusion Matrix Training Set, removed 3 features', errorRateTrainingSet);
% filename_training = 'confmat_training_removed3.png';
% exportgraphics(training_fig,filename_training) 
% 
% 
% errorRateTestSet = calculateErrorRate(confmat_test, length(testSet));
% test_fig = plotConfusionMatrixGPT(confmat_test, 'Confusion Matrix Test Set, removed 3 features',errorRateTestSet);
% filename_test = 'confmat_test_removed3.png';
% exportgraphics(test_fig,filename_test)
% 
% % histogram = plotHistograms(x1all, x2all, x3all, features);
% % exportgraphics(histogram,'histogram.png') 
% 
