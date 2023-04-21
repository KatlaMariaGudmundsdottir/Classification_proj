%% Import data from text file.
% Script for importing data from the following text file:
%
%    C:\Users\svein\Documents\Skole\NTNU\2023 Vår\Estimering\Wovels\vowdata_nohead.dat
%
% To extend the code to different selected data or a different text file, generate a function instead of a script.

% Auto-generated by MATLAB on 2023/04/20 09:46:58
clear all
%% Initialize variables.
filename = 'C:\Users\svein\Documents\Skole\NTNU\2023 Vår\Estimering\Wovels\vowdata_nohead.dat';

%% Format for each line of text:
%   column1: text (%s)
%	column2: double (%f)
%   column3: double (%f)
%	column4: double (%f)
%   column5: double (%f)
%	column6: double (%f)
%   column7: double (%f)
%	column8: double (%f)
%   column9: double (%f)
%	column10: double (%f)
%   column11: double (%f)
%	column12: double (%f)
%   column13: double (%f)
%	column14: double (%f)
%   column15: double (%f)
%	column16: double (%f)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%5s%4f%4f%5f%5f%5f%5f%5f%5f%5f%5f%5f%5f%5f%5f%5f%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to the format.
% This call is based on the structure of the file used to generate this code. If an error occurs for a different file, try regenerating the code from the Import Tool.
dataArray = textscan(fileID, formatSpec, 'Delimiter', '', 'WhiteSpace', '', 'TextType', 'string',  'ReturnOnError', false);

%% Remove white space around all cell columns.
dataArray{1} = strtrim(dataArray{1});

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post processing code is included. To generate code which works for unimportable data, select unimportable cells in a file and regenerate the script.

%% Create output variable
vowdatanohead = table(dataArray{1:end-1}, 'VariableNames', {'identifier','durationMS','f0ss','F1_ss','F2_ss','F3_ss','F4_ss','F1_20','F2_20','F3_20','F1_50','F2_50','F3_50','F1_80','F2_80','F3_80'});

%% Clear temporary variables
clearvars filename formatSpec fileID dataArray ans;
classes = 12;
features = 15;
dataPerClass = 139;
trainingPerClass = 30;
testPerClass = dataPerClass-trainingPerClass;

testSet = zeros(testPerClass*classes,features);
trainingSet = zeros(trainingPerClass*classes,features);
for i = 1:classes
    index1 = trainingPerClass+dataPerClass*(i-1)+1;
    index2 = dataPerClass*i;
    tempTest = vowdatanohead(index1:index2,2:16);
    testSet(1+testPerClass*(i-1):testPerClass*i, 1:features) = table2array(tempTest);

    index1 = 1+dataPerClass*(i-1);
    index2 = trainingPerClass + dataPerClass*(i-1);
    tempTraining = vowdatanohead(index1:index2,2:16);
    trainingSet(1+trainingPerClass*(i-1):trainingPerClass*i, 1:features) = table2array(tempTraining) ;
end 


columnMeans = zeros(classes,features);
covMatrcies = zeros(classes*features,features);
for i = 1:classes
    index1 = 1+trainingPerClass*(i-1);
    index2 = trainingPerClass*(i);
    res = trainingSet(index1:index2,:);
    columnMeans(i,:) = mean(res,1);
    covMatrcies(1+features*(i-1):features*i,:) = cov(res);
end


predictedClasses = zeros(1, length(trainingSet));
for k =  1:length(trainingSet)
    xk = trainingSet(k,:);
    pdf_k = zeros(1,classes);
    for C = 1:classes 
        pdf_k(C) = mvnpdf(xk,columnMeans(classes,:), covMatrcies(1+features*(C-1):features*C,:));
    end
    predictedClasses(k) = find(pdf_k==max(pdf_k));
end

% mvnpdf(138,2:16, )



columnMeans = mean(resArrayAEClass, 1);
covAE = cov(resArrayAEClass);

