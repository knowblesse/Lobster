%% drawGraphs
% load EventClassification result file and draw grphas for figures

load('EventClassificationResult.mat');
numSession = 40;

%% Load Number of Units vs Accuracy Data

% numSession = 40;
% maxUnit = 26;
% numRepeat = 5;
% unitAccuracy = cell(numSession, 2, numRepeat, maxUnit);
% 
% for session = 1 : numel(sessionPaths)
%     TANK_name = cell2mat(sessionPaths{session});
%     TANK_location = char(strcat(basePath, filesep, TANK_name));
%     % Load and process data
%     load(TANK_location);
%     
%     % Result (shuffle/real) x (repeat) x (unit)
%     unitAccuracy(session, :, :, 1 : size(result, 3)) = num2cell(result);    
% end

%% Draw Number of Units vs Accuracy Graph

figure('Position', [680   553   443   425]);
subplot(1,1,1);
hold on;

for session = 1 : numSession
    plot(mean(cell2mat(squeeze(unitAccuracy(session, 1, :, :))), 1), 'Color', xkcd.light_grey);
    plot(mean(cell2mat(squeeze(unitAccuracy(session, 2, :, :))), 1), 'Color', xkcd.light_grey);
end

meanAccuracy = zeros(2, 26);
for i = 1 : 26
    meanAccuracy(1,i) = mean(cell2mat(squeeze(unitAccuracy(:, 1, :, i))), 'all');
    meanAccuracy(2,i) = mean(cell2mat(squeeze(unitAccuracy(:, 2, :, i))), 'all');
end

l1 = plot(meanAccuracy(1,:), 'Color', 'k', 'LineWidth', 2, 'LineStyle', '--');
l2 = plot(meanAccuracy(2,:), 'Color', 'k', 'LineWidth', 2);

xlabel('Number of Units', 'FontName', 'Noto Sans');
ylabel('Balanced accuracy');

xlim([1, 26]);
ylim([0.2, 1]);

legend([l1, l2], {'Shuffled', 'Real'});

set(gca, 'FontName', 'Noto Sans');


%% draw heatmap
cmap_PL = [...
    linspace(1, xkcd.pig_pink(1), 100)',...
    linspace(1, xkcd.pig_pink(2), 100)',...
    linspace(1, xkcd.pig_pink(3), 100)'];
cmap_IL = [...
    linspace(1, xkcd.sky_blue(1), 100)',...
    linspace(1, xkcd.sky_blue(2), 100)',...
    linspace(1, xkcd.sky_blue(3), 100)'];

PLdata = result(contains(tankNames, "PL"));
ILdata = result(contains(tankNames, "IL"));

% PL
cm_shuffled = zeros(numel(PLdata), 3, 3);
cm_real = zeros(numel(PLdata), 3, 3);
for i = 1 : numel(PLdata)
    cm = double(PLdata{i}.confusion_matrix);
    cm_shuffled(i, :, :) = squeeze(cm(1, :, :)) ./ repmat(sum(squeeze(cm(1, :, :)), 2), 1, 3);
    cm_real(i, :, :) = squeeze(cm(2, :, :)) ./ repmat(sum(squeeze(cm(2, :, :)), 2), 1, 3);
end

fig = figure();

subplot(2,2,1);
ax1 = heatmap({'HE', 'AHW', 'EHW'}, {'HE', 'AHW', 'EHW'}, squeeze(mean(cm_shuffled, 1)));
caxis([0, 1]);
colormap(ax1, cmap_PL);
ax1.CellLabelFormat = '%0.2f';
xlabel('Predicted');
ylabel('Actual');
title('Shuffled');

subplot(2,2,2);
ax2 = heatmap({'HE', 'AHW', 'EHW'}, {'HE', 'AHW', 'EHW'}, squeeze(mean(cm_real, 1)));
caxis([0, 1]);
colormap(ax2, cmap_PL);
ax2.CellLabelFormat = '%0.2f';
xlabel('Predicted');
ylabel('Actual');
title('Real');

% IL
cm_shuffled = zeros(numel(ILdata), 3, 3);
cm_real = zeros(numel(ILdata), 3, 3);
for i = 1 : numel(ILdata)
    cm = double(ILdata{i}.confusion_matrix);
    cm_shuffled(i, :, :) = squeeze(cm(1, :, :)) ./ repmat(sum(squeeze(cm(1, :, :)), 2), 1, 3);
    cm_real(i, :, :) = squeeze(cm(2, :, :)) ./ repmat(sum(squeeze(cm(2, :, :)), 2), 1, 3);
end

subplot(2,2,3);
ax3 = heatmap({'HE', 'AHW', 'EHW'}, {'HE', 'AHW', 'EHW'}, squeeze(mean(cm_shuffled, 1)));
caxis([0, 1]);
colormap(ax3, cmap_IL);
ax3.CellLabelFormat = '%0.2f';
xlabel('Predicted');
ylabel('Actual');
title('Shuffled');

subplot(2,2,4);
ax4 = heatmap({'HE', 'AHW', 'EHW'}, {'HE', 'AHW', 'EHW'}, squeeze(mean(cm_real, 1)));
caxis([0, 1]);
ax4.CellLabelFormat = '%0.2f';
colormap(ax4, cmap_IL);
xlabel('Predicted');
ylabel('Actual');
title('Real');

set(gca, 'FontName', 'Noto Sans');