%% SortedPETH_Scripts

output = loadAllUnitData();

%% Responsiveness calculation
zscore_threshold = 4;
bin_size = 80;

valid_IRON_zscores = zeros(size(output,1), bin_size);
first_LICK_zscores = zeros(size(output,1), bin_size);
valid_IROF_zscores = zeros(size(output,1), bin_size);
valid_IROF_A_zscores = zeros(size(output,1), bin_size);
valid_IROF_E_zscores = zeros(size(output,1), bin_size);

responsive_LICK = zeros(size(output,1),1);
responsive_IRON = zeros(size(output,1),1);
responsive_IROF = zeros(size(output,1),3);

for i = 1 : size(output, 1)        
    first_LICK_zscores(i, :) = output.Data{i}.zscore.first_LICK;
    valid_IRON_zscores(i, :) = output.Zscore{i}.valid_IRON;
    valid_IROF_zscores(i, :) = output.Zscore{i}.valid_IROF;
    valid_IROF_A_zscores(i, :) = output.Zscore{i}.valid_IROF_A;
    valid_IROF_E_zscores(i, :) = output.Zscore{i}.valid_IROF_E;

    responsive_LICK(i) = any(abs(first_LICK_zscores(i, :)) > zscore_threshold);
    
    responsive_IRON(i) = any(abs(valid_IRON_zscores(i, :)) > zscore_threshold);
    
    responsive_IROF(i,1) = any(abs(valid_IROF_zscores(i, :)) > zscore_threshold);
    responsive_IROF(i,2) = any(abs(valid_IROF_A_zscores(i, :)) > zscore_threshold);
    responsive_IROF(i,3) = any(abs(valid_IROF_E_zscores(i, :)) > zscore_threshold);
end

fprintf('LK Responsive unit : %.2f %%\n', sum(responsive_LICK) / size(output,1)*100);
fprintf('HE Responsive unit : %.2f %%\n', sum(responsive_IRON) / size(output,1)*100);
fprintf('HW Responsive unit : %.2f %%\n', sum(responsive_IROF(:,1)) / size(output,1)*100);
fprintf('   Avoid Responsive : %.2f %% Escape Responsive : %.2f %%\n\n', sum(responsive_IROF(:,2)) / size(output,1)*100, sum(responsive_IROF(:,3)) / size(output,1)*100);

%clearvars -except output valid* responsive* first*

%% Draw Peak Sorted PETH - HE vs HW

figure('Name', 'SortedPETH_HE_HW', 'Position', [475, 312, 571, 589]);
ax_hm1 = subplot(4,2,[1,3,5]);
ax_hist1 = subplot(4,2,7);
drawPeakSortedPETH(valid_IRON_zscores(logical(responsive_IRON), :), [-2000, 2000], 50, ax_hm1, ax_hist1)

ax_hm2 = subplot(4,2,[2,4,6]);
ax_hist2 = subplot(4,2,8);
drawPeakSortedPETH(valid_IROF_zscores(logical(responsive_IROF), :), [-2000, 2000], 50, ax_hm2, ax_hist2)


%% Draw Peak Sorted PETH - AHW vs EHW
figure('Name', 'SortedPETH_AHW_EHW', 'Position', [475, 312, 571, 589]);
ax_hm1 = subplot(4,2,[1,3,5]);
ax_hist1 = subplot(4,2,7);
drawPeakSortedPETH(valid_IROF_A_zscores(logical(responsive_IROF(:,2)), :), [-2000, 2000], 50, ax_hm1, ax_hist1)

ax_hm2 = subplot(4,2,[2,4,6]);
ax_hist2 = subplot(4,2,8);
drawPeakSortedPETH(valid_IROF_E_zscores(logical(responsive_IROF(:,3)), :), [-2000, 2000], 50, ax_hm2, ax_hist2)

%% Draw Peak Sorted PETH - Lick vs AHW vs EHW
figureSize = [89, 248, 288, 689];
figure('Name', 'SortedPETH_Lick', 'Position', figureSize);
ax_hm1 = subplot(4,1,1:3);
ax_hist1 = subplot(4,1,4);
drawPeakSortedPETH(first_LICK_zscores(logical(responsive_LICK), :), [-2000, 2000], 50, ax_hm1, ax_hist1, 'Name', 'First Lick');
ax_hm1.Clipping = 'off';
hold(ax_hm1, 'on');
fill(ax_hm1, [80, 80, 85, 85], [1, 139, 139, 1], 'r', 'FaceAlpha', 0.3, 'LineStyle', 'None');
fill(ax_hm1, [80, 80, 85, 85], [139, 163, 163, 139], 'y', 'FaceAlpha', 0.3, 'LineStyle', 'None');
fill(ax_hm1, [80, 80, 85, 85], [163, 229, 229, 163], 'b', 'FaceAlpha', 0.3, 'LineStyle', 'None');

figure('Name', 'SortedPETH_AHW', 'Position', figureSize);
ax_hm1 = subplot(4,1,1:3);
ax_hist1 = subplot(4,1,4);
drawPeakSortedPETH(valid_IROF_A_zscores(logical(responsive_IROF(:,2)), :), [-2000, 2000], 50, ax_hm1, ax_hist1, 'Name', 'AHW');
ax_hm1.Clipping = 'off';
hold(ax_hm1, 'on');
fill(ax_hm1, [80, 80, 85, 85], [1, 73, 73, 1], 'r', 'FaceAlpha', 0.3, 'LineStyle', 'None');
fill(ax_hm1, [80, 80, 85, 85], [73, 104, 104, 73], 'y', 'FaceAlpha', 0.3, 'LineStyle', 'None');
fill(ax_hm1, [80, 80, 85, 85], [104, 159, 159, 104], 'b', 'FaceAlpha', 0.3, 'LineStyle', 'None');

figure('Name', 'SortedPETH_EHW', 'Position', figureSize);
ax_hm1 = subplot(4,1,1:3);
ax_hist1 = subplot(4,1,4);
drawPeakSortedPETH(valid_IROF_E_zscores(logical(responsive_IROF(:,3)), :), [-2000, 2000], 50, ax_hm1, ax_hist1, 'Name', 'EHW');
ax_hm1.Clipping = 'off';
hold(ax_hm1, 'on');
fill(ax_hm1, [82, 82, 85, 85], [1, 36, 36, 1], xkcd.red,'LineStyle', 'None');
fill(ax_hm1, [82, 82, 85, 85], [36, 160, 160, 36], xkcd.orange, 'LineStyle', 'None');
fill(ax_hm1, [82, 82, 85, 85], [160, 215, 215, 160], xkcd.blue, 'LineStyle', 'None');

%% How many cells are aligned to the center
bin_size = 80;
bin_center_size = 2;

data = first_LICK_zscores(logical(responsive_LICK), :);
[~, peak_index] = max(data, [], 2);
histcounts(peak_index, [1, bin_size/2-bin_center_size+1, bin_size/2+bin_center_size+1, 80])

data = valid_IROF_A_zscores(logical(responsive_IROF(:,2)), :);
[~, peak_index] = max(data, [], 2);
histcounts(peak_index, [1, bin_size/2-bin_center_size+1, bin_size/2+bin_center_size+1, 80])

data = valid_IROF_E_zscores(logical(responsive_IROF(:,3)), :);
[~, peak_index] = max(data, [], 2);
histcounts(peak_index, [1, bin_size/2-bin_center_size+1, bin_size/2+bin_center_size+1, 80])

% two middle bins are selected as 'Peak near event units'

















