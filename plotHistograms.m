function fig = plotHistograms(x1all, x2all, x3all, features)
    fig = figure;
    til = tiledlayout(4,2,'Padding','Compact');
    for i = 1:4
        nexttile(til,[2,1])
        histogram(x1all(:,i), 'Normalization', 'probability', 'FaceColor', 'b'); hold on;
        histogram(x2all(:,i), 'Normalization', 'probability', 'FaceColor', 'g');
        histogram(x3all(:,i), 'Normalization', 'probability', 'FaceColor', 'r');
        xlabel(sprintf('%s (cm)',char(features(i))));
        ylabel('Probability');
        title(sprintf('Histogram of %s by class', char(features(i))));
%         legend('setosa', 'versicolor', 'virginica');
    end
    leg = legend('setosa', 'versicolor', 'virginica', 'Orientation', 'horizontal');
    leg.Layout.Tile = 'south';
end