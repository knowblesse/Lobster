%% SortedPETH_Scripts
% Scripts for drawing and refining Peak Sorted PETH
load('..\AllUnitData.mat')
%output = loadAllUnitData();
output_PL = output(output.Area == "PL", :);
output_IL = output(output.Area == "IL", :);
addpath('..\FeatureImportance\');
ClassifyUnits;
loadxkcd;

%% Responsiveness calculation

unitData = output;
%unitData = output_IL;

zscore_threshold = 4;
bin_size = 80;
first_LICK_zscores = zeros(size(unitData,1), bin_size);
first_LICK_A_zscores = zeros(size(unitData,1), bin_size);
first_LICK_E_zscores = zeros(size(unitData,1), bin_size);
valid_IROF_zscores = zeros(size(unitData,1), bin_size);
valid_IROF_A_zscores = zeros(size(unitData,1), bin_size);
valid_IROF_E_zscores = zeros(size(unitData,1), bin_size);

responsive = zeros(size(unitData,1),6);

for i = 1 : size(unitData, 1)
    first_LICK_zscores(i,:) = unitData.Zscore{i}.first_LICK;
    first_LICK_A_zscores(i, :) = unitData.Zscore{i}.first_LICK_A;
    first_LICK_E_zscores(i, :) = unitData.Zscore{i}.first_LICK_E;
    valid_IROF_zscores(i,:) = unitData.Zscore{i}.valid_IROF;
    valid_IROF_A_zscores(i, :) = unitData.Zscore{i}.valid_IROF_A;
    valid_IROF_E_zscores(i, :) = unitData.Zscore{i}.valid_IROF_E;

    responsive(i,1) = any(abs(first_LICK_zscores(i,:)) > zscore_threshold);
    responsive(i,2) = any(abs(first_LICK_A_zscores(i, :)) > zscore_threshold);
    responsive(i,3) = any(abs(first_LICK_E_zscores(i, :)) > zscore_threshold);
    responsive(i,4) = any(abs(valid_IROF_zscores(i,:)) > zscore_threshold);
    responsive(i,5) = any(abs(valid_IROF_A_zscores(i, :)) > zscore_threshold);
    responsive(i,6) = any(abs(valid_IROF_E_zscores(i, :)) > zscore_threshold);
end
fprintf('General Responsive : %.2f %%\n', sum(any(responsive,2)) / size(unitData,1) * 100);
fprintf('----------------------------------\n')
fprintf('LK Responsive : %.2f %%\n', sum(responsive(:,1)) / size(unitData,1)  *100);
fprintf('ALK Responsive : %.2f %%\n', sum(responsive(:,2)) / size(unitData,1)  *100);
fprintf('ELK Responsive : %.2f %%\n', sum(responsive(:,3)) / size(unitData,1)  *100);
fprintf('HW Responsive : %.2f %%\n', sum(responsive(:,4)) / size(unitData, 1) * 100);
fprintf('AHW Responsive : %.2f %%\n', sum(responsive(:,5)) / size(unitData, 1) * 100);
fprintf('EHW Responsive : %.2f %%\n', sum(responsive(:,6)) / size(unitData, 1) * 100);

clearvars resp_*

%% Draw Peak Sorted PETH - Lick (Head Entry), Head Withdrawal
figureSize = [89, 248, 288, 689];

figure('Name', 'SortedPETH_HE', 'Position', figureSize);
ax_hm1 = subplot(4,1,1:3);
ax_hist1 = subplot(4,1,4);
sortOrder_HE = drawPeakSortedPETH(first_LICK_zscores(logical(responsive(:,1)),:), [-2000, 2000], 50, ax_hm1, ax_hist1, 'Name', 'Head Entry');
ax_hm1.Clipping = 'off';
ylim(ax_hist1, [-.3, 3]);
p = ylabel('Z');
p.Position(1) = -4;

hold(ax_hm1, 'on');
responsiveUnits = Unit.Group_HE(logical(responsive(:,1)));
PETH_units = responsiveUnits(sortOrder_HE);

colors = [xkcd.algae; xkcd.orange];

for i = 1 : numel(responsiveUnits)
    if PETH_units(i) == 0
        continue;
    end
    line(ax_hm1, [82, 85], [i, i], 'Color', colors(PETH_units(i), :), 'LineWidth',1.8);
end

saveas(gcf, 'C:\Users\Knowblesse\Desktop\1.svg', 'svg');

figure('Name', 'SortedPETH_HW', 'Position', figureSize);
ax_hm1 = subplot(4,1,1:3);
ax_hist1 = subplot(4,1,4);
sortOrder_HW = drawPeakSortedPETH(valid_IROF_zscores(logical(responsive(:,4)),:), [-2000, 2000], 50, ax_hm1, ax_hist1, 'Name', 'Head Withdrawal');
ax_hm1.Clipping = 'off';
ylim(ax_hist1, [-.3, 3]);
p = ylabel('Z');
p.Position(1) = -4;

hold(ax_hm1, 'on');
responsiveUnits = Unit.Group_HW(logical(responsive(:,4)));
PETH_units = responsiveUnits(sortOrder_HW);

colors = [xkcd.red; xkcd.golden_rod; xkcd.blue];

for i = 1 : numel(responsiveUnits)
    if PETH_units(i) == 0
        continue;
    end
    line(ax_hm1, [82, 85], [i, i], 'Color', colors(PETH_units(i), :), 'LineWidth',1.8);
end

saveas(gcf, 'C:\Users\Knowblesse\Desktop\2.svg', 'svg');

figure('Name', 'SortedPETH_AHW', 'Position', figureSize);
ax_hm1 = subplot(4,1,1:3);
ax_hist1 = subplot(4,1,4);
drawPeakSortedPETH(...
    valid_IROF_A_zscores(logical(responsive(:,4)),:), [-2000, 2000], 50, ax_hm1, ax_hist1, ...
    'Name', 'AHW',...
    'ManualIndex',sortOrder_HW);
ax_hm1.Clipping = 'off';
ylim(ax_hist1, [-.3, 3]);
p = ylabel('Z');
p.Position(1) = -4;
saveas(gcf, 'C:\Users\Knowblesse\Desktop\3.svg', 'svg');

figure('Name', 'SortedPETH_EHW', 'Position', figureSize);
ax_hm1 = subplot(4,1,1:3);
ax_hist1 = subplot(4,1,4);
drawPeakSortedPETH(valid_IROF_E_zscores(logical(responsive(:,4)),:), [-2000, 2000], 50, ax_hm1, ax_hist1,...
    'Name', 'EHW',...
    'ManualIndex',sortOrder_HW);
ax_hm1.Clipping = 'off';
ylim(ax_hist1, [-.3, 3]);
p = ylabel('Z');
p.Position(1) = -4;
saveas(gcf, 'C:\Users\Knowblesse\Desktop\4.svg', 'svg');