function fig = plotConfusionMatrix(C, titleString, errorRate)
    % Create a heatmap
    fig = figure;
    imagesc(C); % Display the matrix as an image
    title(titleString);
    xlabel('Predicted Label', 'FontSize', 14);
    ylabel('True Label', 'FontSize', 14);
    
    % Add text labels
    textStrings = num2str(C(:),'%d'); % Convert the counts into strings
    textStrings = strtrim(cellstr(textStrings)); % Remove any space padding
    [x,y] = meshgrid(1:size(C,1),1:size(C,2)); % Create x,y coordinates for the strings
    hStrings = text(x(:),y(:),textStrings(:),'horizontalAlignment','center', 'FontSize', 14);
    midValue = mean(get(gca,'CLim')); % Get the middle value of the color range
    textColors = repmat(C(:) > midValue,1,3); % Choose white or black for the text color
    set(hStrings,{'Color'},num2cell(textColors,2)); % Change the text color
    
    % Customize the heatmap
    colormap(flipud(gray)); % Change the color scheme to black and white
    colorbar; % Add a colorbar

    % Set x and y labels to show whole numbers
    xticks(1:size(C,1)); % Set the x ticks to show whole numbers
    yticks(1:size(C,2)); % Set the y ticks to show whole numbers
    set(gca,'FontSize',14); % Increase font size of the axes labels

    % Add error rate text label
    [numRows, numCols] = size(C);
    errorRateStr = sprintf('Error rate: %.2f%%', errorRate*100);
    text(numCols/2, numRows+2, errorRateStr,'clipping', 'off', 'HorizontalAlignment', 'center', 'FontSize', 14);
end
