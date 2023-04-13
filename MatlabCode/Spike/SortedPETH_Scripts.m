%% SortedPETH_Scripts
% Scripts for drawing and refining Peak Sorted PETH
load('..\AllUnitData.mat')
%output = loadAllUnitData();
output_PL = output(output.Area == "PL", :);
output_IL = output(output.Area == "IL", :);
addpath('..\FeatureImportance\');
ClassifyUnits;

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


%% Draw Unit Class Composition
fig = figure('Name', 'Unit Type Composition', 'Position', [321   542   765   307]); 

ax = subplot(1,1,1);
bplot = bar([...
    sum(output.first_LICK_A_type == 1),...
    sum(output.first_LICK_A_type == 2),...
    sum(output.first_LICK_A_type == 3);...
    sum(output.first_LICK_E_type == 1),...
    sum(output.first_LICK_E_type == 2),...
    sum(output.first_LICK_E_type == 3);...
    sum(output.valid_IROF_A_type == 1),...
    sum(output.valid_IROF_A_type == 2),...
    sum(output.valid_IROF_A_type == 3);...
    sum(output.valid_IROF_E_type == 1),...
    sum(output.valid_IROF_E_type == 2),...
    sum(output.valid_IROF_E_type == 3)],...               
    'FaceColor', 'flat',...
    'LineStyle', 'none');
bplot(1).CData = [xkcd.red; xkcd.red; xkcd.red; xkcd.red];
bplot(2).CData = [xkcd.goldenrod; xkcd.goldenrod; xkcd.goldenrod; xkcd.goldenrod];
bplot(3).CData = [xkcd.blue; xkcd.blue; xkcd.blue; xkcd.blue];

hold on;

text(bplot(1).XEndPoints(1), bplot(1).YData(1) / 2,...
    strcat("n=", num2str(sum(output.first_LICK_A_type == 1))),...
    'Rotation', 90,...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');
text(bplot(2).XEndPoints(1), bplot(2).YData(1) / 2,...
    strcat("n=", num2str(sum(output.first_LICK_A_type == 2))),...
    'Rotation', 90,...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');
text(bplot(3).XEndPoints(1), bplot(3).YData(1) / 2,...
    strcat("n=", num2str(sum(output.first_LICK_A_type == 3))),...
    'Rotation', 90,...
    'Color', 'w',...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');

text(bplot(1).XEndPoints(2), bplot(1).YData(2) / 2,...
    strcat("n=", num2str(sum(output.first_LICK_E_type == 1))),...
    'Rotation', 90,...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');
text(bplot(2).XEndPoints(2), bplot(2).YData(2) / 2,...
    strcat("n=", num2str(sum(output.first_LICK_E_type == 2))),...
    'Rotation', 90,...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');
text(bplot(3).XEndPoints(2), bplot(3).YData(2) / 2,...
    strcat("n=", num2str(sum(output.first_LICK_E_type == 3))),...
    'Rotation', 90,...
    'Color', 'w',...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');

text(bplot(1).XEndPoints(3), bplot(1).YData(3) / 2,...
    strcat("n=", num2str(sum(output.valid_IROF_A_type == 1))),...
    'Rotation', 90,...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');
text(bplot(2).XEndPoints(3), bplot(2).YData(3) / 2,...
    strcat("n=", num2str(sum(output.valid_IROF_A_type == 2))),...
    'Rotation', 90,...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');
text(bplot(3).XEndPoints(3), bplot(3).YData(3) / 2,...
    strcat("n=", num2str(sum(output.valid_IROF_A_type == 3))),...
    'Rotation', 90,...
    'Color', 'w',...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');

text(bplot(1).XEndPoints(4), bplot(1).YData(4) / 2,...
    strcat("n=", num2str(sum(output.valid_IROF_E_type == 1))),...
    'Rotation', 90,...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');
text(bplot(2).XEndPoints(4), bplot(2).YData(4) / 2,...
    strcat("n=", num2str(sum(output.valid_IROF_E_type == 2))),...
    'Rotation', 90,...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');
text(bplot(3).XEndPoints(4), bplot(3).YData(4) / 2,...
    strcat("n=", num2str(sum(output.valid_IROF_E_type == 3))),...
    'Color', 'w',...
    'Rotation', 90,...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');

xticklabels(["AHE", "EHE", "AHW", "EHW"]);
ylim([0, 150]);
ylabel('Units');
legend({'pre-event', 'peri-event', 'post-event'}, 'Location', 'northeast', 'FontName', 'Noto Sans');



%% Class change between avoidance and escape trial

cmap_HE = [...
    linspace(1, xkcd.pig_pink(1), 100)',...
    linspace(1, xkcd.pig_pink(2), 100)',...
    linspace(1, xkcd.pig_pink(3), 100)'];
cmap_HW = [...
    linspace(1, xkcd.sky_blue(1), 100)',...
    linspace(1, xkcd.sky_blue(2), 100)',...
    linspace(1, xkcd.sky_blue(3), 100)'];

% Head Entry
fig = figure();

hm_data = zeros(4,4);
for avoid_class = 0 : 3
    for escape_class = 0 : 3
        hm_data(avoid_class + 1, escape_class + 1) = ...
            sum(all([output.first_LICK_A_type == avoid_class, output.first_LICK_E_type == escape_class], 2))...
                / sum(output.first_LICK_A_type == avoid_class);
    end
end

% change Non-reponsive to the last index
hm_data = hm_data(:, [2,3,4,1]);
hm_data = hm_data([2,3,4,1], :);


% Head Withdrawal
subplot(1,2,1);
ax1 = heatmap({'Pre', 'Peri', 'Post', 'Non'}, {'Pre', 'Peri', 'Post', 'Non'}, hm_data);
caxis([0, 1]);
colormap(ax1, cmap_HE);
ax1.CellLabelFormat = '%0.2f';
xlabel('Escape Class');
ylabel('Avoidance Class');
title('Head Entry');

hm_data = zeros(4,4);
for avoid_class = 0 : 3
    for escape_class = 0 : 3
        hm_data(avoid_class + 1, escape_class + 1) = ...
            sum(all([output.valid_IROF_A_type == avoid_class, output.valid_IROF_E_type == escape_class], 2))...
                / sum(output.valid_IROF_A_type == avoid_class);
    end
end

% rearrange the matrix so the Non-reponsive units are located to the last index
hm_data = hm_data(:, [2,3,4,1]);
hm_data = hm_data([2,3,4,1], :);

subplot(1,2,2);
ax2 = heatmap({'Pre', 'Peri', 'Post', 'Non'}, {'Pre', 'Peri', 'Post', 'Non'}, hm_data);
caxis([0, 1]);
colormap(ax2, cmap_HW);
ax2.CellLabelFormat = '%0.2f';
xlabel('Escape Class');
ylabel('Avoidance Class');
title('Head Withdrawal');


%% Sorted PETH of AHW Class 1 units
figureSize = [89, 248, 288, 689];

figure('Name', 'SortedPETH_AHW_AHWC1', 'Position', figureSize);
ax_hm1 = subplot(4,1,1:3);
ax_hist1 = subplot(4,1,4);
drawPeakSortedPETH(valid_IROF_A_zscores(output.valid_IROF_A_type == 1, :), [-2000, 2000], 50, ax_hm1, ax_hist1, 'Name', 'AHW-AHWC1');
ax_hm1.Clipping = 'off';
ylim(ax_hist1, [-2, 2]);

figure('Name', 'SortedPETH_EHW_AHWC1', 'Position', figureSize);
ax_hm2 = subplot(4,1,1:3);
ax_hist2 = subplot(4,1,4);
drawPeakSortedPETH(valid_IROF_E_zscores(output.valid_IROF_A_type == 1, :), [-2000, 2000], 50, ax_hm2, ax_hist2, 'Name', 'EHW-AHWC1', 'Sort', false);
ax_hm2.Clipping = 'off';
ylim(ax_hist2, [-2, 2]);

