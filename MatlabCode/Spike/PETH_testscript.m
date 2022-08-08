%% Draw PETH
index = 24 ;
event = 'valid_IRON'; % ["TRON","first_IRON","valid_IRON","first_LICK","valid_IROF","ATTK","TROF"]
TIMEWINDOW = [-1000, 1000];

numTrial = size(output.Data{index}.binned_spike.TRON, 1);
spikes = cell(numTrial,1);
ParsedData = BehavDataParser(strcat('D:\Data\Lobster\Lobster_Recording-200319-161008\', ...
    output.Subject{index}, filesep,...
    output.Session{index}));
timepoint = getTimepointFromParsedData(ParsedData);

for trial = 1 : numTrial
    spikes{trial} = output.RawSpikeData{index}(...
        and(output.RawSpikeData{index} >= timepoint.(event)(trial) + TIMEWINDOW(1), output.RawSpikeData{index} < timepoint.(event)(trial) + TIMEWINDOW(2)))...
        - timepoint.(event)(trial);
end

fig = figure('Name', strcat("Index : ", num2str(index), " event : ", event));
clf;
axis_ = drawPETH(spikes, TIMEWINDOW);
axis_{1}.Parent = fig;
axis_{2}.Parent = fig;
subplot(3,1,1:2);
title(strcat(output.Session{index}, '-', num2str(output.Cell(index))), 'Interpreter', 'none');
subplot(3,1,3);
hold on;
yyaxis right;
plot(linspace(-1000+(2000/80), 1000-(2000/80), 40), output.Data{index}.zscore.(event),'Color', 'r', 'LineWidth', 2, 'LineStyle', '-');


%% Draw Sorted PETH

TIMEWINDOW_LEFT = -1000; %(ms)
TIMEWINDOW_RIGHT = +1000; %(ms)
TIMEWINDOW_BIN = 50; %(ms) 
numBin = (TIMEWINDOW_RIGHT - TIMEWINDOW_LEFT)/TIMEWINDOW_BIN; % number of bins

title_text = 'Head Entry (responsive)';

binnedZ = valid_IRON_zscores(logical(responsive_IRON), :);

[~, i] =max(binnedZ,[],2);
[~, ix] = sort(i);

new_result = binnedZ(ix,:);

figure('Position', [-1410, 112, 560, 828]);
ax1 = subplot(4,7,[2:4, 9:11, 16:18]);
imagesc(new_result);
hold on;
line([20.5,20.5],[1,size(binnedZ,1)], 'Color', 'w', 'LineWidth', 1); 
xticklabels({'-1000', '-500', '0', '+500', '+1000'});
xticks(0.5 : 10 : numBin + 0.5);
xlim([0.5,40.5]);
yticks([]);
title(title_text);
colormap 'jet';
caxis([-5, 20]);
set(gca, 'FontName', 'Noto Sans');
colorbar;
drawnow;

% Peak Histogram
[~,i] = max(binnedZ,[],2);
ax2 = subplot(4,7,[23:25]);
histogram(i,(1:numBin)-0.5, 'FaceColor','k','LineStyle','none');
hold on;
ylim_ = ylim();
line([20,20],ylim_, 'Color', 'r', 'LineWidth', 1); 
xlim([0.5, 39.5]);
ylim(ylim_);
xticks([0.5, 10, 20, 30, 39.5]);
xticklabels({'-1000', '-500', '0', '+500', '+1000'});
ylabel('Number of Unit');
xlabel('Time (ms)');
title('Peak Histogram');
ax2.Position(3) = ax1.Position(3);
set(gca, 'FontName', 'Noto Sans');

subplot(4,7,[1,8,15]);

scores = importance_score(logical(responsive_IRON), 2);

[~, i] =max(binnedZ,[],2);
[~, ix] = sort(i);

plot(movmean(scores(ix),10), 'k');

set(gca, 'View', [90, 90])
xlim([1, size(binnedZ, 1)])
title('Unit Importance');
set(gca, 'FontName', 'Noto Sans');


binnedZ = valid_IROF_zscores(logical(responsive_IRON), :);
new_result = binnedZ(ix,:);

ax3 = subplot(4,7,[5:7, 12:14, 19:21]);
imagesc(new_result);
hold on;
line([20.5,20.5],[1,size(binnedZ,1)], 'Color', 'w', 'LineWidth', 1); 
xticklabels({'-1000', '-500', '0', '+500', '+1000'});
xticks(0.5 : 10 : numBin + 0.5);
xlim([0.5,40.5]);
yticks([]);
title('Corresponding HW');
colormap 'jet';
caxis([-5, 20]);
set(gca, 'FontName', 'Noto Sans');
colorbar;
drawnow;

% Peak Histogram
[~,i] = max(binnedZ,[],2);
ax4 = subplot(4,7,26:28);
histogram(i,(1:numBin)-0.5, 'FaceColor','k','LineStyle','none');
hold on;
ylim_ = ylim();
line([20,20],ylim_, 'Color', 'r', 'LineWidth', 1); 
xlim([0.5, 39.5]);
ylim(ylim_);
xticks([0.5, 10, 20, 30, 39.5]);
xticklabels({'-1000', '-500', '0', '+500', '+1000'});
ylabel('Number of Unit');
xlabel('Time (ms)');
title('Peak Histogram');
ax4.Position(3) = ax3.Position(3);
set(gca, 'FontName', 'Noto Sans');




%% Aligned to HW


TIMEWINDOW_LEFT = -1000; %(ms)
TIMEWINDOW_RIGHT = +1000; %(ms)
TIMEWINDOW_BIN = 50; %(ms) 
numBin = (TIMEWINDOW_RIGHT - TIMEWINDOW_LEFT)/TIMEWINDOW_BIN; % number of bins

title_text = 'Head Withdrawal (responsive)';

binnedZ = valid_IROF_zscores(logical(responsive_IROF), :);

[~, i] =max(binnedZ,[],2);
[~, ix] = sort(i);

new_result = binnedZ(ix,:);

figure('Position', [-1410, 112, 560, 828]);
ax1 = subplot(4,7,[2:4, 9:11, 16:18]);
imagesc(new_result);
hold on;
line([20.5,20.5],[1,size(binnedZ,1)], 'Color', 'w', 'LineWidth', 1); 
xticklabels({'-1000', '-500', '0', '+500', '+1000'});
xticks(0.5 : 10 : numBin + 0.5);
xlim([0.5,40.5]);
yticks([]);
title(title_text);
colormap 'jet';
caxis([-5, 20]);
set(gca, 'FontName', 'Noto Sans');
colorbar;
drawnow;

% Peak Histogram
[~,i] = max(binnedZ,[],2);
ax2 = subplot(4,7,[23:25]);
histogram(i,(1:numBin)-0.5, 'FaceColor','k','LineStyle','none');
hold on;
ylim_ = ylim();
line([20,20],ylim_, 'Color', 'r', 'LineWidth', 1); 
xlim([0.5, 39.5]);
ylim(ylim_);
xticks([0.5, 10, 20, 30, 39.5]);
xticklabels({'-1000', '-500', '0', '+500', '+1000'});
ylabel('Number of Unit');
xlabel('Time (ms)');
title('Peak Histogram');
ax2.Position(3) = ax1.Position(3);
set(gca, 'FontName', 'Noto Sans');

subplot(4,7,[1,8,15]);

scores = importance_score(logical(responsive_IROF), 2);

[~, i] =max(binnedZ,[],2);
[~, ix] = sort(i);

plot(movmean(scores(ix),10), 'k');

set(gca, 'View', [90, 90])
xlim([1, size(binnedZ, 1)])
title('Unit Importance');
set(gca, 'FontName', 'Noto Sans');


binnedZ = valid_IRON_zscores(logical(responsive_IROF), :);
new_result = binnedZ(ix,:);

ax3 = subplot(4,7,[5:7, 12:14, 19:21]);
imagesc(new_result);
hold on;
line([20.5,20.5],[1,size(binnedZ,1)], 'Color', 'w', 'LineWidth', 1); 
xticklabels({'-1000', '-500', '0', '+500', '+1000'});
xticks(0.5 : 10 : numBin + 0.5);
xlim([0.5,40.5]);
yticks([]);
title('Corresponding HE');
colormap 'jet';
caxis([-5, 20]);
set(gca, 'FontName', 'Noto Sans');
colorbar;
drawnow;

% Peak Histogram
[~,i] = max(binnedZ,[],2);
ax4 = subplot(4,7,26:28);
histogram(i,(1:numBin)-0.5, 'FaceColor','k','LineStyle','none');
hold on;
ylim_ = ylim();
line([20,20],ylim_, 'Color', 'r', 'LineWidth', 1); 
xlim([0.5, 39.5]);
ylim(ylim_);
xticks([0.5, 10, 20, 30, 39.5]);
xticklabels({'-1000', '-500', '0', '+500', '+1000'});
ylabel('Number of Unit');
xlabel('Time (ms)');
title('Peak Histogram');
ax4.Position(3) = ax3.Position(3);
set(gca, 'FontName', 'Noto Sans');







