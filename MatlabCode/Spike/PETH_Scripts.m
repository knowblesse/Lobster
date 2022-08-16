%% PETH_Scripts
output = loadAllUnitData();

%% Draw Single PETH
index = 15 ;
event = 'valid_IROF'; % ["TRON","first_IRON","valid_IRON","first_LICK","valid_IROF","ATTK","TROF"]
TIMEWINDOW = [-1000, 1000];

timepoint = getTimepointFromParsedData(output.BehavData{index});

% Generate spikes cell to feed in to the drawPETH function
numTrial = size(timepoint.TRON, 1);
spikes = cell(numTrial,1);
for trial = 1 : numTrial
    spikes{trial} = output.RawSpikeData{index}(...
        and(output.RawSpikeData{index} >= timepoint.(event)(trial) + TIMEWINDOW(1), output.RawSpikeData{index} < timepoint.(event)(trial) + TIMEWINDOW(2)))...
        - timepoint.(event)(trial);
end

fig = figure(...
    'Name', strcat("Index : ", num2str(index)," event : ", event),...
    'Position', [1094, 592, 560, 301]...
    );
clf;
ax_raster1 = subplot(3,1,1:2);
title(strcat(output.Session{index}, '-', num2str(output.Cell(index))), 'Interpreter', 'none');
ax_histo1 = subplot(3,1,3);
drawPETH(spikes, TIMEWINDOW, ax_raster1, ax_histo1, false);
clearvars -except output

%% Draw Representative A/E HW PETH
TIMEWINDOW = [-1000, 1000];
for index = [72, 516]
    %nLoad BehaviorData and Load Timepoints
    numTrial = size(output.Data{index}.binned_spike.TRON, 1);
    ParsedData = output.BehavData{index};
    behaviorResult = analyticValueExtractor(ParsedData, false, true);
    timepoint = getTimepointFromParsedData(ParsedData);

    % Divide Avoid and Escape PETH
    event = 'valid_IROF';
    spikes = cell(numTrial,1);
    for trial = 1 : numTrial
        spikes{trial} = output.RawSpikeData{index}(...
            and(output.RawSpikeData{index} >= timepoint.(event)(trial) + TIMEWINDOW(1), output.RawSpikeData{index} < timepoint.(event)(trial) + TIMEWINDOW(2)))...
            - timepoint.(event)(trial);
    end

    fig = figure(...
        'Name', strcat("Index : ", num2str(index), " event : Head Withdrawal"),...
        'Position', [1094, 592, 560, 301]);
    clf;
    ax_raster1 = subplot(3,2,[1,3]);
    title("Avoid", 'FontName', 'Noto Sans');
    ax_histo1 = subplot(3,2,5);
    behaviorIndex = behaviorResult == 'A';
    drawPETH(spikes(behaviorIndex, :), TIMEWINDOW, ax_raster1, ax_histo1, true);

    ax_raster2 = subplot(3,2,[2,4]);
    title("Escape", 'FontName', 'Noto Sans');
    ax_histo2 = subplot(3,2,6);
    behaviorIndex = behaviorResult == 'E';
    drawPETH(spikes(behaviorIndex, :), TIMEWINDOW, ax_raster2, ax_histo2, true);
    
    yl1 = ylim(ax_histo1);
    yl2 = ylim(ax_histo2);
    
    maxlim = max(yl1(2), yl2(2));
    ylim(ax_histo1, [0, maxlim]);
    ylim(ax_histo2, [0, maxlim]);
end
clearvars -except output


%% Responsiveness calculation
zscore_threshold = 4;
valid_IRON_zscores = zeros(size(output,1), 40);
valid_IROF_zscores = zeros(size(output,1), 40);
valid_IROF_A_zscores = zeros(size(output,1), 40);
valid_IROF_E_zscores = zeros(size(output,1), 40);
responsive_IRON = zeros(size(output,1),1);
responsive_IROF = zeros(size(output,1),3);
for i = 1 : size(output, 1)        
    valid_IRON_zscores(i, :) = output.Zscore{i}.valid_IRON;
    valid_IROF_zscores(i, :) = output.Zscore{i}.valid_IROF;
    valid_IROF_A_zscores(i, :) = output.Zscore{i}.valid_IROF_A;
    valid_IROF_E_zscores(i, :) = output.Zscore{i}.valid_IROF_E;
    
    responsive_IRON(i) = any(abs(valid_IRON_zscores(i, :)) > zscore_threshold);
    
    responsive_IROF(i,1) = any(abs(valid_IROF_zscores(i, :)) > zscore_threshold);
    responsive_IROF(i,2) = any(abs(valid_IROF_A_zscores(i, :)) > zscore_threshold);
    responsive_IROF(i,3) = any(abs(valid_IROF_E_zscores(i, :)) > zscore_threshold);
end

fprintf('HE Responsive unit : %.2f %%\n', sum(responsive_IRON) / size(output,1)*100);
fprintf('HW Responsive unit : %.2f %%\n', sum(responsive_IROF(:,1)) / size(output,1)*100);
fprintf('   Avoid Responsive : %.2f %% Escape Responsive : %.2f %%\n\n', sum(responsive_IROF(:,2)) / size(output,1)*100, sum(responsive_IROF(:,3)) / size(output,1)*100);

clearvars -except output valid* responsive*

%% Draw Peak Sorted PETH - HE vs HW

TIMEWINDOW = [-1000, 1000]; % (ms)
TIMEWINDOW_BIN = 50; %(ms) 
numBin = diff(TIMEWINDOW)/TIMEWINDOW_BIN; % number of bins

% Create sorted Z score - Data to draw

zs = valid_IRON_zscores(logical(responsive_IRON), :);
[~, peak_loc_valid_IRON] = max(zs, [], 2);
[~, ix] = sort(peak_loc_valid_IRON);
sorted_valid_IRON = zs(ix, :);

zs = valid_IROF_zscores(logical(responsive_IROF(:,1)), :);
[~, peak_loc_valid_IROF] = max(zs, [], 2);
[~, ix] = sort(peak_loc_valid_IROF);
sorted_valid_IROF = zs(ix, :);

clearvars zs ix

figure('Name', 'SortedPETH_HE_HW', 'Position', [475, 312, 571, 589]);

hmap1 = subplot(4,2,[1,3,5]);
imagesc(hmap1, sorted_valid_IRON);
hold on;
line([20.5,20.5],[1,size(sorted_valid_IRON,1)], 'Color', 'w', 'LineWidth', 1); 
xticklabels({'-1000', '-500', '0', '+500', '+1000'});
xticks(0.5 : 10 : numBin + 0.5);
xlim([0.5,40.5]);
ylabel('Unit');
title('Head Entry');
colormap 'jet';
caxis([-5, 20]);
set(gca, 'FontName', 'Noto Sans');
drawnow;

hist1 = subplot(4,2,7);
histogram(peak_loc_valid_IRON,(1:numBin)-0.5, 'FaceColor','k','LineStyle','none');
hold on;
ylim_ = ylim();
line([20,20],ylim_, 'Color', 'r', 'LineWidth', 1); 
xlim([0.5, 39.5]);
ylim(ylim_);
xticks([0.5, 10, 20, 30, 39.5]);
xticklabels({'-1000', '-500', '0', '+500', '+1000'});
ylabel('Number of Unit');
xlabel('Time (ms)');
hist1.Position(3) = hmap1.Position(3);
set(gca, 'FontName', 'Noto Sans');

hmap2 = subplot(4,2,[2,4,6]);
imagesc(hmap2, sorted_valid_IROF);
hold on;
line([20.5,20.5],[1,size(sorted_valid_IROF,1)], 'Color', 'w', 'LineWidth', 1); 
xticklabels({'-1000', '-500', '0', '+500', '+1000'});
xticks(0.5 : 10 : numBin + 0.5);
xlim([0.5,40.5]);
ylabel('Unit');
title('Head Withdrawal');
colormap 'jet';
caxis([-5, 20]);
set(gca, 'FontName', 'Noto Sans');
% When drawing figure, uncomment this line, save figure, import
% colorbar, and plot it next to this subplot. 
% colorbar; 
drawnow;

hist2 = subplot(4,2,8);
histogram(peak_loc_valid_IROF,(1:numBin)-0.5, 'FaceColor','k','LineStyle','none');
hold on;
ylim_ = ylim();
line([20,20],ylim_, 'Color', 'r', 'LineWidth', 1); 
xlim([0.5, 39.5]);
ylim(ylim_);
xticks([0.5, 10, 20, 30, 39.5]);
xticklabels({'-1000', '-500', '0', '+500', '+1000'});
ylabel('Number of Unit');
xlabel('Time (ms)');
hist2.Position(3) = hmap2.Position(3);
set(gca, 'FontName', 'Noto Sans');

%% Draw Peak Sorted PETH - AHW vs EHW

TIMEWINDOW = [-1000, 1000]; % (ms)
TIMEWINDOW_BIN = 50; %(ms) 
numBin = diff(TIMEWINDOW)/TIMEWINDOW_BIN; % number of bins

% Create sorted Z score - Data to draw

zs = valid_IROF_A_zscores(logical(responsive_IROF(:,2)), :);
[~, peak_loc_valid_IROF_A] = max(zs, [], 2);
[~, ix] = sort(peak_loc_valid_IROF_A);
sorted_valid_IROF_A = zs(ix, :);

zs = valid_IROF_E_zscores(logical(responsive_IROF(:,3)), :);
[~, peak_loc_valid_IROF_E] = max(zs, [], 2);
[~, ix] = sort(peak_loc_valid_IROF_E);
sorted_valid_IROF_E = zs(ix, :);

clearvars zs ix

figure('Name', 'SortedPETH_AHW_EHW', 'Position', [475, 312, 571, 589]);

hmap1 = subplot(4,2,[1,3,5]);
imagesc(hmap1, sorted_valid_IROF_A);
hold on;
line([20.5,20.5],[1,size(sorted_valid_IROF_A,1)], 'Color', 'w', 'LineWidth', 1); 
xticklabels({'-1000', '-500', '0', '+500', '+1000'});
xticks(0.5 : 10 : numBin + 0.5);
xlim([0.5,40.5]);
ylabel('Unit');
title('Avoidance Head Withdrawal');
colormap 'jet';
caxis([-5, 20]);
set(gca, 'FontName', 'Noto Sans');
drawnow;

hist1 = subplot(4,2,7);
histogram(peak_loc_valid_IROF_A,(1:numBin)-0.5, 'FaceColor','k','LineStyle','none');
hold on;
ylim_ = ylim();
line([20,20],ylim_, 'Color', 'r', 'LineWidth', 1); 
xlim([0.5, 39.5]);
ylim(ylim_);
xticks([0.5, 10, 20, 30, 39.5]);
xticklabels({'-1000', '-500', '0', '+500', '+1000'});
ylabel('Number of Unit');
xlabel('Time (ms)');
hist1.Position(3) = hmap1.Position(3);
set(gca, 'FontName', 'Noto Sans');

hmap2 = subplot(4,2,[2,4,6]);
imagesc(hmap2, sorted_valid_IROF_E);
hold on;
line([20.5,20.5],[1,size(sorted_valid_IROF_E,1)], 'Color', 'w', 'LineWidth', 1); 
xticklabels({'-1000', '-500', '0', '+500', '+1000'});
xticks(0.5 : 10 : numBin + 0.5);
xlim([0.5,40.5]);
ylabel('Unit');
title('Escape Head Withdrawal');
colormap 'jet';
caxis([-5, 20]);
set(gca, 'FontName', 'Noto Sans');
% When drawing figure, uncomment this line, save figure, import
% colorbar, and plot it next to this subplot. 
% colorbar; 
drawnow;

hist2 = subplot(4,2,8);
histogram(peak_loc_valid_IROF_E,(1:numBin)-0.5, 'FaceColor','k','LineStyle','none');
hold on;
ylim_ = ylim();
line([20,20],ylim_, 'Color', 'r', 'LineWidth', 1); 
xlim([0.5, 39.5]);
ylim(ylim_);
xticks([0.5, 10, 20, 30, 39.5]);
xticklabels({'-1000', '-500', '0', '+500', '+1000'});
ylabel('Number of Unit');
xlabel('Time (ms)');
hist2.Position(3) = hmap2.Position(3);
set(gca, 'FontName', 'Noto Sans');
