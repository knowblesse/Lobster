%% SortedPETH_Scripts
% Scripts for drawing and refining Peak Sorted PETH
load('C:\Users\Knowblesse\SynologyDrive\AllUnitData.mat')
%output = loadAllUnitData();
output_PL = output(output.Area == "PL", :);
output_IL = output(output.Area == "IL", :);

%% Responsiveness calculation

unitData = output;

zscore_threshold = 4;
bin_size = 80;

first_LICK_A_zscores = zeros(size(unitData,1), bin_size);
first_LICK_E_zscores = zeros(size(unitData,1), bin_size);
valid_IROF_A_zscores = zeros(size(unitData,1), bin_size);
valid_IROF_E_zscores = zeros(size(unitData,1), bin_size);

responsive = zeros(size(unitData,1),4);

for i = 1 : size(unitData, 1)        
    first_LICK_A_zscores(i, :) = unitData.Zscore{i}.first_LICK_A;
    first_LICK_E_zscores(i, :) = unitData.Zscore{i}.first_LICK_E;
    valid_IROF_A_zscores(i, :) = unitData.Zscore{i}.valid_IROF_A;
    valid_IROF_E_zscores(i, :) = unitData.Zscore{i}.valid_IROF_E;

    responsive(i,1) = any(abs(first_LICK_A_zscores(i, :)) > zscore_threshold);
    responsive(i,2) = any(abs(first_LICK_E_zscores(i, :)) > zscore_threshold);
    responsive(i,3) = any(abs(valid_IROF_A_zscores(i, :)) > zscore_threshold);
    responsive(i,4) = any(abs(valid_IROF_E_zscores(i, :)) > zscore_threshold);
end

fprintf('ALK Responsive : %.2f %%\n', sum(responsive(:,1)) / size(unitData,1)  *100);
fprintf('ELK Responsive : %.2f %%\n', sum(responsive(:,2)) / size(unitData,1)  *100);
fprintf('AHW Responsive : %.2f %%\n', sum(responsive(:,3)) / size(unitData, 1) * 100);
fprintf('EHW Responsive : %.2f %%\n', sum(responsive(:,4)) / size(unitData, 1) * 100);

%% Gather units into 3 groups and label them
bin_size = 80; % 2 sec with 50ms bin size
bin_center_size = 2; % 2 bins around the onset of the event (= 50ms * 2 = 100ms around the event = 200ms)
% this makes a unit with peak at 39, 40, 41, 42 bin classified into the second group
% 1 : Pre Event
% 2 : Near Event
% 3 : Post Event

data = first_LICK_A_zscores(logical(responsive(:,1)), :);
[~, peak_index_LICK_A] = max(data, [], 2); 
[first_LICK_A_type_count, ~, first_LICK_A_type] = histcounts(peak_index_LICK_A, [1, bin_size/2-bin_center_size+1, bin_size/2+bin_center_size+1, 80]); 
first_LICK_A_type_range = cumsum(first_LICK_A_type_count); 
array2append = zeros(size(responsive,1),1);
array2append(logical(responsive(:,1))) = first_LICK_A_type; 
output = [output, table(array2append, 'VariableNames', "first_LICK_A_type")]; 

data = first_LICK_E_zscores(logical(responsive(:,1)), :);
[~, peak_index_LICK_E] = max(data, [], 2);
[first_LICK_E_type_count, ~, first_LICK_E_type] = histcounts(peak_index_LICK_E, [1, bin_size/2-bin_center_size+1, bin_size/2+bin_center_size+1, 80]);
first_LICK_E_type_range = cumsum(first_LICK_E_type_count);
array2append = zeros(size(responsive,1),1);
array2append(logical(responsive(:,1))) = first_LICK_E_type;
output = [output, table(array2append, 'VariableNames', "first_LICK_E_type")];

data = valid_IROF_A_zscores(logical(responsive(:,2)), :);
[~, peak_index_IROF_A] = max(data, [], 2);
[valid_IROF_A_type_count, ~, valid_IROF_A_type] = histcounts(peak_index_IROF_A, [1, bin_size/2-bin_center_size+1, bin_size/2+bin_center_size+1, 80]);
valid_IROF_A_type_range = cumsum(valid_IROF_A_type_count);
array2append = zeros(size(responsive,1),1);
array2append(logical(responsive(:,2))) = valid_IROF_A_type;
output = [output, table(array2append, 'VariableNames', "valid_IROF_A_type")];

data = valid_IROF_E_zscores(logical(responsive(:,3)), :);
[~, peak_index_IROF_E] = max(data, [], 2);
[valid_IROF_E_type_count, ~, valid_IROF_E_type] = histcounts(peak_index_IROF_E, [1, bin_size/2-bin_center_size+1, bin_size/2+bin_center_size+1, 80]);
valid_IROF_E_type_range = cumsum(valid_IROF_E_type_count);
array2append = zeros(size(responsive,1),1);
array2append(logical(responsive(:,3))) = valid_IROF_E_type;
output = [output, table(array2append, 'VariableNames', "valid_IROF_E_type")];

%% Draw Peak Sorted PETH - Lick vs AHW vs EHW
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

figure('Name', 'SortedPETH_ELick', 'Position', figureSize);
ax_hm1 = subplot(4,1,1:3);
ax_hist1 = subplot(4,1,4);
drawPeakSortedPETH(first_LICK_E_zscores(logical(responsive(:,1)), :), [-2000, 2000], 50, ax_hm1, ax_hist1, 'Name', 'First Lick');
ax_hm1.Clipping = 'off';
hold(ax_hm1, 'on');
fill(ax_hm1, [82, 82, 85, 85], [1, first_LICK_E_type_range(1), first_LICK_E_type_range(1), 1], xkcd.red, 'LineStyle', 'None');
fill(ax_hm1, [82, 82, 85, 85], [first_LICK_E_type_range(1), first_LICK_E_type_range(2), first_LICK_E_type_range(2), first_LICK_E_type_range(1)], xkcd.goldenrod, 'LineStyle', 'None');
fill(ax_hm1, [82, 82, 85, 85], [first_LICK_E_type_range(2), first_LICK_E_type_range(3), first_LICK_E_type_range(3), first_LICK_E_type_range(2)], xkcd.blue, 'LineStyle', 'None');

figure('Name', 'SortedPETH_AHW', 'Position', figureSize);
ax_hm1 = subplot(4,1,1:3);
ax_hist1 = subplot(4,1,4);
drawPeakSortedPETH(valid_IROF_A_zscores(logical(responsive(:,2)), :), [-2000, 2000], 50, ax_hm1, ax_hist1, 'Name', 'AHW');
ax_hm1.Clipping = 'off';
hold(ax_hm1, 'on');
fill(ax_hm1, [82, 82, 85, 85], [1, valid_IROF_A_type_range(1), valid_IROF_A_type_range(1), 1], xkcd.red, 'LineStyle', 'None');
fill(ax_hm1, [82, 82, 85, 85], [valid_IROF_A_type_range(1), valid_IROF_A_type_range(2), valid_IROF_A_type_range(2), valid_IROF_A_type_range(1)], xkcd.goldenrod, 'LineStyle', 'None');
fill(ax_hm1, [82, 82, 85, 85], [valid_IROF_A_type_range(2), valid_IROF_A_type_range(3), valid_IROF_A_type_range(3), valid_IROF_A_type_range(2)], xkcd.blue, 'LineStyle', 'None');

figure('Name', 'SortedPETH_EHW', 'Position', figureSize);
ax_hm1 = subplot(4,1,1:3);
ax_hist1 = subplot(4,1,4);
drawPeakSortedPETH(valid_IROF_E_zscores(logical(responsive(:,3)), :), [-2000, 2000], 50, ax_hm1, ax_hist1, 'Name', 'EHW');
ax_hm1.Clipping = 'off';
hold(ax_hm1, 'on');
fill(ax_hm1, [82, 82, 85, 85], [1, valid_IROF_E_type_range(1), valid_IROF_E_type_range(1), 1], xkcd.red,'LineStyle', 'None');
fill(ax_hm1, [82, 82, 85, 85], [valid_IROF_E_type_range(1), valid_IROF_E_type_range(2), valid_IROF_E_type_range(2), valid_IROF_E_type_range(1)], xkcd.goldenrod, 'LineStyle', 'None');
fill(ax_hm1, [82, 82, 85, 85], [valid_IROF_E_type_range(2), valid_IROF_E_type_range(3), valid_IROF_E_type_range(3), valid_IROF_E_type_range(2)], xkcd.blue, 'LineStyle', 'None');

%% Draw Composition Graph
fig = figure('Name', 'Unit Composition', 'Position', [321   546   896   303]); 

% Index
plIndex = output.Area == "PL";
ilIndex = output.Area == "IL";
% Total Responsive
totUnit_PL = sum(plIndex);
totUnit_IL = sum(ilIndex);

resUnit_PL = sum(any(responsive(plIndex, :), 2));
resUnit_IL = sum(any(responsive(ilIndex, :), 2));

ax1 = subplot(1,8,1:2);
bplot = bar([resUnit_PL / totUnit_PL, (totUnit_PL - resUnit_PL) / totUnit_PL;...
             resUnit_IL / totUnit_IL, (totUnit_IL - resUnit_IL) / totUnit_IL],...
             'stacked',...
             'FaceColor', 'flat',...
             'LineStyle', 'none');
bplot(1).CData = [xkcd.pig_pink; xkcd.sky_blue];
bplot(2).CData = [xkcd.grey; xkcd.grey];
hold on;
text(1, resUnit_PL / totUnit_PL / 2, ...
    [num2str(resUnit_PL / totUnit_PL, '%.2f'),strcat("n=", num2str(resUnit_PL))],...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');
text(1, resUnit_PL / totUnit_PL + (totUnit_PL - resUnit_PL) / totUnit_PL / 2, ...
    [num2str((totUnit_PL - resUnit_PL) / totUnit_PL, '%.2f'),strcat("n=", num2str((totUnit_PL - resUnit_PL)))],...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');
text(2, resUnit_IL / totUnit_IL / 2, ...
    [num2str(resUnit_IL / totUnit_IL, '%.2f'),strcat("n=", num2str(resUnit_IL))],...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');
text(2, resUnit_IL / totUnit_IL + (totUnit_IL - resUnit_IL) / totUnit_IL / 2, ...
    [num2str((totUnit_IL - resUnit_IL) / totUnit_IL, '%.2f'),strcat("n=", num2str((totUnit_IL - resUnit_IL)))],...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');
xticklabels(["PL", "IL"]);
ylabel('Proportions');
set(ax1, 'FontName', 'Noto Sans');
set(ax1, 'FontSize', 9);


% Event Responsive
data2plot = [sum(responsive(plIndex,1)); sum(responsive(plIndex,2)); sum(responsive(plIndex,3))];
data2plot = [data2plot, resUnit_PL - data2plot] ./ resUnit_PL;

ax2 = subplot(1,8,3:5);
bplot = bar(data2plot,...
             'stacked',...
             'FaceColor', 'flat',...
             'LineStyle', 'none');
bplot(1).CData = repmat(xkcd.pig_pink, 3, 1);
bplot(2).CData = repmat(xkcd.pig_pink, 3, 1);
bplot(2).FaceAlpha = 0.2;
hold on;
text(1, data2plot(1,1) / 2, ...
    [num2str(data2plot(1,1), '%.2f'),strcat("n=", num2str(sum(responsive(plIndex,1))))],...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');
text(2, data2plot(2,1) / 2, ...
    [num2str(data2plot(2,1), '%.2f'),strcat("n=", num2str(sum(responsive(plIndex,2))))],...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');
text(3, data2plot(3,1) / 2, ...
    [num2str(data2plot(3,1), '%.2f'),strcat("n=", num2str(sum(responsive(plIndex,3))))],...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');

xticklabels(["HE", "AHW", "EHW"]);
yticklabels({});
set(ax2, 'FontName', 'Noto Sans');
set(ax2, 'FontSize', 9);

% Event Responsive
data2plot = [sum(responsive(ilIndex,1)); sum(responsive(ilIndex,2)); sum(responsive(ilIndex,3))];
data2plot = [data2plot, resUnit_IL - data2plot] ./ resUnit_IL;

ax3 = subplot(1,8,6:8);
bplot = bar(data2plot,...
             'stacked',...
             'FaceColor', 'flat',...
             'LineStyle', 'none');
bplot(1).CData = repmat(xkcd.sky_blue, 3, 1);
bplot(2).CData = repmat(xkcd.sky_blue, 3, 1);
bplot(2).FaceAlpha = 0.2;
hold on;
text(1, data2plot(1,1) / 2, ...
    [num2str(data2plot(1,1), '%.2f'),strcat("n=", num2str(sum(responsive(ilIndex,1))))],...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');
text(2, data2plot(2,1) / 2, ...
    [num2str(data2plot(2,1), '%.2f'),strcat("n=", num2str(sum(responsive(ilIndex,2))))],...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');
text(3, data2plot(3,1) / 2, ...
    [num2str(data2plot(3,1), '%.2f'),strcat("n=", num2str(sum(responsive(ilIndex,3))))],...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');

xticklabels(["HE", "AHW", "EHW"]);
yticklabels({});
set(ax3, 'FontName', 'Noto Sans');
set(ax3, 'FontSize', 9);


%% Draw Unit Type Composition
fig = figure('Name', 'Unit Type Composition', 'Position', [321   542   765   307]); 

ax = subplot(1,1,1);
bplot = bar([...
    sum(output.first_LICK_type == 1),...
    sum(output.first_LICK_type == 2),...
    sum(output.first_LICK_type == 3);...
    sum(output.valid_IROF_A_type == 1),...
    sum(output.valid_IROF_A_type == 2),...
    sum(output.valid_IROF_A_type == 3);...
    sum(output.valid_IROF_E_type == 1),...
    sum(output.valid_IROF_E_type == 2),...
    sum(output.valid_IROF_E_type == 3)],...               
    'FaceColor', 'flat',...
    'LineStyle', 'none');
bplot(1).CData = [xkcd.red; xkcd.red; xkcd.red];
bplot(2).CData = [xkcd.goldenrod; xkcd.goldenrod; xkcd.goldenrod];
bplot(3).CData = [xkcd.blue; xkcd.blue; xkcd.blue];

hold on;
text(bplot(1).XEndPoints(1), bplot(1).YData(1) / 2,...
    strcat("n=", num2str(sum(output.first_LICK_type == 1))),...
    'Rotation', 90,...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');
text(bplot(2).XEndPoints(1), bplot(2).YData(1) / 2,...
    strcat("n=", num2str(sum(output.first_LICK_type == 2))),...
    'Rotation', 90,...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');
text(bplot(3).XEndPoints(1), bplot(3).YData(1) / 2,...
    strcat("n=", num2str(sum(output.first_LICK_type == 3))),...
    'Rotation', 90,...
    'Color', 'w',...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');

text(bplot(1).XEndPoints(2), bplot(1).YData(2) / 2,...
    strcat("n=", num2str(sum(output.valid_IROF_A_type == 1))),...
    'Rotation', 90,...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');
text(bplot(2).XEndPoints(2), bplot(2).YData(2) / 2,...
    strcat("n=", num2str(sum(output.valid_IROF_A_type == 2))),...
    'Rotation', 90,...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');
text(bplot(3).XEndPoints(2), bplot(3).YData(2) / 2,...
    strcat("n=", num2str(sum(output.valid_IROF_A_type == 3))),...
    'Rotation', 90,...
    'Color', 'w',...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');

text(bplot(1).XEndPoints(3), bplot(1).YData(3) / 2,...
    strcat("n=", num2str(sum(output.valid_IROF_E_type == 1))),...
    'Rotation', 90,...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');
text(bplot(2).XEndPoints(3), bplot(2).YData(3) / 2,...
    strcat("n=", num2str(sum(output.valid_IROF_E_type == 2))),...
    'Rotation', 90,...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');
text(bplot(3).XEndPoints(3), bplot(3).YData(3) / 2,...
    strcat("n=", num2str(sum(output.valid_IROF_E_type == 3))),...
    'Color', 'w',...
    'Rotation', 90,...
    'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'middle',...
    'FontName', 'Noto Sans');

xticklabels(["HE", "AHW", "EHW"]);
ylim([0, 150]);
ylabel('Units');
legend({'pre-event', 'peri-event', 'post-event'}, 'Location', 'northeast', 'FontName', 'Noto Sans');
set(ax2, 'FontName', 'Noto Sans');
set(ax2, 'FontSize', 9);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data Processing Scripts                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load Feature Importance Data and save into the data table
load('C:\VCF\Lobster\MatlabCode\Spike\Output.mat');
r_ = [];
rhe_ = [];
rahw_ = [];
rehw_ = [];
for i = 1 : 40
    r_ = [r_; mean(result{i}.importance_score,1)'];
    rhe_ = [rhe_; mean(result{i}.importance_score_HE,1)'];
    rahw_ = [rahw_; mean(result{i}.importance_score_AHW,1)'];
    rehw_ = [rehw_; mean(result{i}.importance_score_EHW,1)'];
end
output = [output, table(r_, 'VariableNames', "FI")];
output = [output, table(rhe_, 'VariableNames', "FI_HE")];
output = [output, table(rahw_, 'VariableNames', "FI_AHW")];
output = [output, table(rehw_, 'VariableNames', "FI_EHW")];

%% Feature Importance on Balanced accuracy
output.FI(output.first_LICK_type == 1) * 100
output.FI(output.first_LICK_type == 2) * 100
output.FI(output.first_LICK_type == 3) * 100
output.FI(output.first_LICK_type == 0) * 100

output.FI(output.valid_IROF_A_type == 1) * 100
output.FI(output.valid_IROF_A_type == 2) * 100
output.FI(output.valid_IROF_A_type == 3) * 100
output.FI(output.valid_IROF_A_type == 0) * 100

output.FI(output.valid_IROF_E_type == 1) * 100
output.FI(output.valid_IROF_E_type == 2) * 100
output.FI(output.valid_IROF_E_type == 3) * 100
output.FI(output.valid_IROF_E_type == 0) * 100

%% Feature Importance on the accuracy of the each class
output.FI_HE(output.first_LICK_type == 1) * 100
output.FI_HE(output.first_LICK_type == 2) * 100
output.FI_HE(output.first_LICK_type == 3) * 100
output.FI_HE(output.first_LICK_type == 0) * 100

output.FI_AHW(output.valid_IROF_A_type == 1) * 100
output.FI_AHW(output.valid_IROF_A_type == 2) * 100
output.FI_AHW(output.valid_IROF_A_type == 3) * 100
output.FI_AHW(output.valid_IROF_A_type == 0) * 100

output.FI_EHW(output.valid_IROF_E_type == 1) * 100
output.FI_EHW(output.valid_IROF_E_type == 2) * 100
output.FI_EHW(output.valid_IROF_E_type == 3) * 100
output.FI_EHW(output.valid_IROF_E_type == 0) * 100
