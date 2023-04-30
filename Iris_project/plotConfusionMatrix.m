function fig = plotConfusionMatrix(C, titleString, errorRate)
    % Create a heatmap
    fig = figure;
    set(fig, 'Units', 'pixels', 'Position', [0, 0, 560, 480]); % Set figure size
    
    % Add the main plot
    mainPlot = subplot('Position', [0, 0, 1, 1]);
    imagesc(mainPlot,C); % Display the matrix as an image
    title(mainPlot,titleString);
    xlabel(mainPlot,'Predicted Label', 'FontSize', 14);
    ylabel(mainPlot,'True Label', 'FontSize', 14);
    
    % Add text labels
    textStrings = num2str(C(:),'%d'); % Convert the counts into strings
    textStrings = strtrim(cellstr(textStrings)); % Remove any space padding
    [x,y] = meshgrid(1:size(C,1),1:size(C,2)); % Create x,y coordinates for the strings
    hStrings = text(mainPlot,x(:),y(:),textStrings(:),'horizontalAlignment','center', 'FontSize', 14);
    midValue = mean(get(mainPlot,'CLim')); % Get the middle value of the color range
    textColors = repmat(C(:) > midValue,1,3); % Choose white or black for the text color
    set(hStrings,{'Color'},num2cell(textColors,2)); % Change the text color
    
    % Customize the heatmap
%     whiteGreen = [linspace(1,0,256)', linspace(1,0.5,256)', linspace(1,0.2,256)'];
    colormap(mainPlot,flipud(gray)); % Change the color scheme to black and white
    colorbar(mainPlot); % Add a colorbar
    
    % Set x and y labels to show whole numbers
    xticks(mainPlot,1:size(C,1)); % Set the x ticks to show whole numbers
    yticks(mainPlot,1:size(C,2)); % Set the y ticks to show whole numbers
    set(mainPlot,'FontSize',14); % Increase font size of the axes labels

    axpos = mainPlot.Position;
    subpos = axpos + [0.1, 0.2, -0.22, -0.3]; % Add space for the x-axis label
    mainPlot.Position = subpos;

    % Add error rate text label
%     [numRows, numCols] = size(C);
%     axpos = fig.Position;
%     subpos = axpos + [0, 0, 0, 60]; % Add space for the x-axis label
%     set(fig, 'Units', 'pixels', 'Position', subpos); % Set figure size
% 
%     axpos = fig.Position

%     set(mainPlot, 'Position', [0, 0.1, 0.9, 0.8]);
%     % Add the subplot
    subplot('Position',[0 0 1 0.05]);
    set(gca, 'Visible', 'off');
    errorRateStr = sprintf('Error rate: %.2f%%', errorRate*100);
    text(0.5, 0.5, errorRateStr,'clipping', 'off', 'HorizontalAlignment', 'center', 'FontSize', 12);
end
