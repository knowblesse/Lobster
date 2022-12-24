%% SortedPETH_Scripts
% Scripts for drawing and refining Peak Sorted PETH
load('C:\Users\Knowblesse\SynologyDrive\AllUnitData.mat')
%output = loadAllUnitData();
output_PL = output(output.Area == "PL", :);
output_IL = output(output.Area == "IL", :);

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

%% Gather units into 3 groups and label them
bin_size = 80; % 2 sec with 50ms bin size
bin_center_size = 2; % 2 bins around the onset of the event (= 50ms * 2 = 100ms around the event = 200ms)
% this makes a unit with peak at 39, 40, 41, 42 bin classified into the second group
% 1 : Pre Event 0.4875
% 2 : Near Event 0.025
% 3 : Post Event 0.4875

data = first_LICK_zscores(logical(responsive(:,1)), :);
[~, peak_index_LICK] = max(data, [], 2); 
[first_LICK_type_count, ~, first_LICK_type] = histcounts(peak_index_LICK, [1, bin_size/2-bin_center_size+1, bin_size/2+bin_center_size+1, 80]); 
first_LICK_type_range = cumsum(first_LICK_type_count); 
array2append = zeros(size(responsive,1),1);
array2append(logical(responsive(:,1))) = first_LICK_type; 
output = [output, table(array2append, 'VariableNames', "first_LICK_type")]; 

data = first_LICK_A_zscores(logical(responsive(:,2)), :);
[~, peak_index_LICK_A] = max(data, [], 2); 
[first_LICK_A_type_count, ~, first_LICK_A_type] = histcounts(peak_index_LICK_A, [1, bin_size/2-bin_center_size+1, bin_size/2+bin_center_size+1, 80]); 
first_LICK_A_type_range = cumsum(first_LICK_A_type_count); 
array2append = zeros(size(responsive,1),1);
array2append(logical(responsive(:,2))) = first_LICK_A_type; 
output = [output, table(array2append, 'VariableNames', "first_LICK_A_type")]; 

data = first_LICK_E_zscores(logical(responsive(:,3)), :);
[~, peak_index_LICK_E] = max(data, [], 2);
[first_LICK_E_type_count, ~, first_LICK_E_type] = histcounts(peak_index_LICK_E, [1, bin_size/2-bin_center_size+1, bin_size/2+bin_center_size+1, 80]);
first_LICK_E_type_range = cumsum(first_LICK_E_type_count);
array2append = zeros(size(responsive,1),1);
array2append(logical(responsive(:,3))) = first_LICK_E_type;
output = [output, table(array2append, 'VariableNames', "first_LICK_E_type")];

data = valid_IROF_zscores(logical(responsive(:,4)), :);
[~, peak_index_IROF] = max(data, [], 2);
[valid_IROF_type_count, ~, valid_IROF_type] = histcounts(peak_index_IROF, [1, bin_size/2-bin_center_size+1, bin_size/2+bin_center_size+1, 80]);
valid_IROF_type_range = cumsum(valid_IROF_type_count);
array2append = zeros(size(responsive,1),1);
array2append(logical(responsive(:,4))) = valid_IROF_type;
output = [output, table(array2append, 'VariableNames', "valid_IROF_type")];

data = valid_IROF_A_zscores(logical(responsive(:,5)), :);
[~, peak_index_IROF_A] = max(data, [], 2);
[valid_IROF_A_type_count, ~, valid_IROF_A_type] = histcounts(peak_index_IROF_A, [1, bin_size/2-bin_center_size+1, bin_size/2+bin_center_size+1, 80]);
valid_IROF_A_type_range = cumsum(valid_IROF_A_type_count);
array2append = zeros(size(responsive,1),1);
array2append(logical(responsive(:,5))) = valid_IROF_A_type;
output = [output, table(array2append, 'VariableNames', "valid_IROF_A_type")];

data = valid_IROF_E_zscores(logical(responsive(:,6)), :);
[~, peak_index_IROF_E] = max(data, [], 2);
[valid_IROF_E_type_count, ~, valid_IROF_E_type] = histcounts(peak_index_IROF_E, [1, bin_size/2-bin_center_size+1, bin_size/2+bin_center_size+1, 80]);
valid_IROF_E_type_range = cumsum(valid_IROF_E_type_count);
array2append = zeros(size(responsive,1),1);
array2append(logical(responsive(:,6))) = valid_IROF_E_type;
output = [output, table(array2append, 'VariableNames', "valid_IROF_E_type")];

%% Draw Peak Sorted PETH - A/E Lick, A/E Head Withdrawal
figureSize = [89, 248, 288, 689];

figure('Name', 'SortedPETH_ALick', 'Position', figureSize);
ax_hm1 = subplot(4,1,1:3);
ax_hist1 = subplot(4,1,4);
drawPeakSortedPETH(first_LICK_A_zscores(logical(responsive(:,1)), :), [-2000, 2000], 50, ax_hm1, ax_hist1, 'Name', 'First Lick');
ax_hm1.Clipping = 'off';
hold(ax_hm1, 'on');
fill(ax_hm1, [82, 82, 85, 85], [1, first_LICK_A_type_range(1), first_LICK_A_type_range(1), 1], xkcd.red, 'LineStyle', 'None');
fill(ax_hm1, [82, 82, 85, 85], [first_LICK_A_type_range(1), first_LICK_A_type_range(2), first_LICK_A_type_range(2), first_LICK_A_type_range(1)], xkcd.goldenrod, 'LineStyle', 'None');
fill(ax_hm1, [82, 82, 85, 85], [first_LICK_A_type_range(2), first_LICK_A_type_range(3), first_LICK_A_type_range(3), first_LICK_A_type_range(2)], xkcd.blue, 'LineStyle', 'None');
ylim(ax_hist1, [-.3, 2]);
p = ylabel('Z');
p.Position(1) = -4;
saveas(gcf, 'C:\Users\Knowblesse\Desktop\1.svg', 'svg');

figure('Name', 'SortedPETH_ELick', 'Position', figureSize);
ax_hm1 = subplot(4,1,1:3);
ax_hist1 = subplot(4,1,4);
drawPeakSortedPETH(first_LICK_E_zscores(logical(responsive(:,1)), :), [-2000, 2000], 50, ax_hm1, ax_hist1, 'Name', 'First Lick');
ax_hm1.Clipping = 'off';
hold(ax_hm1, 'on');
fill(ax_hm1, [82, 82, 85, 85], [1, first_LICK_E_type_range(1), first_LICK_E_type_range(1), 1], xkcd.red, 'LineStyle', 'None');
fill(ax_hm1, [82, 82, 85, 85], [first_LICK_E_type_range(1), first_LICK_E_type_range(2), first_LICK_E_type_range(2), first_LICK_E_type_range(1)], xkcd.goldenrod, 'LineStyle', 'None');
fill(ax_hm1, [82, 82, 85, 85], [first_LICK_E_type_range(2), first_LICK_E_type_range(3), first_LICK_E_type_range(3), first_LICK_E_type_range(2)], xkcd.blue, 'LineStyle', 'None');
ylim(ax_hist1, [-.3, 2]);
p = ylabel('Z');
p.Position(1) = -4;
saveas(gcf, 'C:\Users\Knowblesse\Desktop\2.svg', 'svg');

figure('Name', 'SortedPETH_AHW', 'Position', figureSize);
ax_hm1 = subplot(4,1,1:3);
ax_hist1 = subplot(4,1,4);
drawPeakSortedPETH(valid_IROF_A_zscores(logical(responsive(:,2)), :), [-2000, 2000], 50, ax_hm1, ax_hist1, 'Name', 'AHW');
ax_hm1.Clipping = 'off';
hold(ax_hm1, 'on');
fill(ax_hm1, [82, 82, 85, 85], [1, valid_IROF_A_type_range(1), valid_IROF_A_type_range(1), 1], xkcd.red, 'LineStyle', 'None');
fill(ax_hm1, [82, 82, 85, 85], [valid_IROF_A_type_range(1), valid_IROF_A_type_range(2), valid_IROF_A_type_range(2), valid_IROF_A_type_range(1)], xkcd.goldenrod, 'LineStyle', 'None');
fill(ax_hm1, [82, 82, 85, 85], [valid_IROF_A_type_range(2), valid_IROF_A_type_range(3), valid_IROF_A_type_range(3), valid_IROF_A_type_range(2)], xkcd.blue, 'LineStyle', 'None');
ylim(ax_hist1, [-.5, .5]);
p = ylabel('Z');
p.Position(1) = -4;
saveas(gcf, 'C:\Users\Knowblesse\Desktop\3.svg', 'svg');

figure('Name', 'SortedPETH_EHW', 'Position', figureSize);
ax_hm1 = subplot(4,1,1:3);
ax_hist1 = subplot(4,1,4);
drawPeakSortedPETH(valid_IROF_E_zscores(logical(responsive(:,3)), :), [-2000, 2000], 50, ax_hm1, ax_hist1, 'Name', 'EHW');
ax_hm1.Clipping = 'off';
hold(ax_hm1, 'on');
fill(ax_hm1, [82, 82, 85, 85], [1, valid_IROF_E_type_range(1), valid_IROF_E_type_range(1), 1], xkcd.red,'LineStyle', 'None');
fill(ax_hm1, [82, 82, 85, 85], [valid_IROF_E_type_range(1), valid_IROF_E_type_range(2), valid_IROF_E_type_range(2), valid_IROF_E_type_range(1)], xkcd.goldenrod, 'LineStyle', 'None');
fill(ax_hm1, [82, 82, 85, 85], [valid_IROF_E_type_range(2), valid_IROF_E_type_range(3), valid_IROF_E_type_range(3), valid_IROF_E_type_range(2)], xkcd.blue, 'LineStyle', 'None');
ylim(ax_hist1, [-.5, 2]);
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

