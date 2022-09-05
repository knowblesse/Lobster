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

figure();
subplot(1,1,1);
hold on;

for session = 1 : numSession
    plot(mean(cell2mat(squeeze(unitAccuracy(session, 1, :, :))), 1), 'Color', xkcd.grey);
    plot(mean(cell2mat(squeeze(unitAccuracy(session, 2, :, :))), 1), 'Color', xkcd.black);
end

meanAccuracy = zeros(2, 26);
for i = 1 : 26
    meanAccuracy(1,i) = mean(cell2mat(squeeze(unitAccuracy(:, 1, :, i))), 'all');
    meanAccuracy(2,i) = mean(cell2mat(squeeze(unitAccuracy(:, 2, :, i))), 'all');
end

plot(meanAccuracy(1,:), 'Color', 'r');
plot(meanAccuracy(2,:), 'Color', 'b');

%% draw heatmap

PLdata = result(contains(tankNames, "PL"));
ILdata = result(contains(tankNames, "IL"));

cm_shuffled = zeros(numel(PLdata), 3, 3);
cm_real = zeros(numel(PLdata), 3, 3);
for i = 1 : numel(PLdata)
    cm = double(PLdata{i}.confusion_matrix);
    cm_shuffled(i, :, :) = squeeze(cm(1, :, :)) ./ repmat(sum(squeeze(cm(1, :, :)), 2), 1, 3);
    cm_real(i, :, :) = squeeze(cm(2, :, :)) ./ repmat(sum(squeeze(cm(2, :, :)), 2), 1, 3);
end

fig = figure('Name', 'PL');

subplot(2,2,1);
heatmap(squeeze(mean(cm_shuffled, 1)));
caxis([0, 1]);

subplot(2,2,2);
heatmap(squeeze(mean(cm_real, 1)));
caxis([0, 1]);



cm_shuffled = zeros(numel(ILdata), 3, 3);
cm_real = zeros(numel(ILdata), 3, 3);
for i = 1 : numel(ILdata)
    cm = double(ILdata{i}.confusion_matrix);
    cm_shuffled(i, :, :) = squeeze(cm(1, :, :)) ./ repmat(sum(squeeze(cm(1, :, :)), 2), 1, 3);
    cm_real(i, :, :) = squeeze(cm(2, :, :)) ./ repmat(sum(squeeze(cm(2, :, :)), 2), 1, 3);
end

subplot(2,2,3);
heatmap(squeeze(mean(cm_shuffled, 1)));
caxis([0, 1]);

subplot(2,2,4);
heatmap(squeeze(mean(cm_real, 1)));
caxis([0, 1]);