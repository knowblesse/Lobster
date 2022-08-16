%% Draw Peak Sorted PETH - HE vs HW (Locked to HE)

TIMEWINDOW = [-1000, 1000]; % (ms)
TIMEWINDOW_BIN = 50; %(ms) 
numBin = diff(TIMEWINDOW)/TIMEWINDOW_BIN; % number of bins

% Create sorted Z score - Data to draw

zs = valid_IRON_zscores(logical(responsive_IRON), :);
[~, peak_loc_valid_IRON] = max(zs, [], 2);
[~, ix] = sort(peak_loc_valid_IRON);
sorted_valid_IRON = zs(ix, :);

zs = valid_IROF_zscores(logical(responsive_IRON(:,1)), :); % Locked to IRON
%[~, peak_loc_valid_IROF] = max(zs, [], 2);
%[~, ix] = sort(peak_loc_valid_IROF);
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

%% Draw Peak Sorted PETH - HE vs HW (Locked to HW)

TIMEWINDOW = [-1000, 1000]; % (ms)
TIMEWINDOW_BIN = 50; %(ms) 
numBin = diff(TIMEWINDOW)/TIMEWINDOW_BIN; % number of bins

% Create sorted Z score - Data to draw

zs = valid_IROF_zscores(logical(responsive_IROF(:,1)), :); % Locked to IROF
[~, peak_loc_valid_IROF] = max(zs, [], 2);
[~, ix] = sort(peak_loc_valid_IROF);
sorted_valid_IROF = zs(ix, :);

zs = valid_IRON_zscores(logical(responsive_IROF(:,1)), :);
%[~, peak_loc_valid_IRON] = max(zs, [], 2);
%[~, ix] = sort(peak_loc_valid_IRON);
sorted_valid_IRON = zs(ix, :);


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

%% Draw Peak Sorted PETH - AHW vs EHW (Locked to AHW)

TIMEWINDOW = [-1000, 1000]; % (ms)
TIMEWINDOW_BIN = 50; %(ms) 
numBin = diff(TIMEWINDOW)/TIMEWINDOW_BIN; % number of bins

% Create sorted Z score - Data to draw

zs = valid_IROF_A_zscores(logical(responsive_IROF(:,2)), :);
[~, peak_loc_valid_IROF_A] = max(zs, [], 2);
[~, ix] = sort(peak_loc_valid_IROF_A);
sorted_valid_IROF_A = zs(ix, :);

zs = valid_IROF_E_zscores(logical(responsive_IROF(:,2)), :); % Locked to AHW
%[~, peak_loc_valid_IROF_E] = max(zs, [], 2);
%[~, ix] = sort(peak_loc_valid_IROF_E);
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

%% Draw Peak Sorted PETH - AHW vs EHW (Locked to EHW)

TIMEWINDOW = [-1000, 1000]; % (ms)
TIMEWINDOW_BIN = 50; %(ms) 
numBin = diff(TIMEWINDOW)/TIMEWINDOW_BIN; % number of bins

% Create sorted Z score - Data to draw
zs = valid_IROF_E_zscores(logical(responsive_IROF(:,3)), :);
[~, peak_loc_valid_IROF_E] = max(zs, [], 2);
[~, ix] = sort(peak_loc_valid_IROF_E);
sorted_valid_IROF_E = zs(ix, :);

zs = valid_IROF_A_zscores(logical(responsive_IROF(:,3)), :); % Locked to EHW
%[~, peak_loc_valid_IROF_A] = max(zs, [], 2);
%[~, ix] = sort(peak_loc_valid_IROF_A);
sorted_valid_IROF_A = zs(ix, :);

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

