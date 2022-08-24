%% SortedPETH_Scripts
load('C:\Users\Knowblesse\SynologyDrive\AllUnitData.mat')
%output = loadAllUnitData();
output_PL = output(output.Area == "PL", :);
output_IL = output(output.Area == "IL", :);

%% Responsiveness calculation

unitData = output;

zscore_threshold = 4;
bin_size = 80;

first_LICK_zscores = zeros(size(unitData,1), bin_size);
valid_IROF_A_zscores = zeros(size(unitData,1), bin_size);
valid_IROF_E_zscores = zeros(size(unitData,1), bin_size);

responsive = zeros(size(unitData,1),3);

for i = 1 : size(unitData, 1)        
    first_LICK_zscores(i, :) = unitData.Data{i}.zscore.first_LICK;
    valid_IROF_A_zscores(i, :) = unitData.Zscore{i}.valid_IROF_A;
    valid_IROF_E_zscores(i, :) = unitData.Zscore{i}.valid_IROF_E;

    responsive(i,1) = any(abs(first_LICK_zscores(i, :)) > zscore_threshold);
    responsive(i,2) = any(abs(valid_IROF_A_zscores(i, :)) > zscore_threshold);
    responsive(i,3) = any(abs(valid_IROF_E_zscores(i, :)) > zscore_threshold);
end

fprintf('LK Responsive  : %.2f %%\n', sum(responsive(:,1)) / size(unitData,1)  *100);
fprintf('AHW Responsive : %.2f %%\n', sum(responsive(:,2)) / size(unitData, 1) * 100);
fprintf('EHW Responsive : %.2f %%\n', sum(responsive(:,3)) / size(unitData, 1) * 100);

%% Venn Diagram
num = 0;
target = [1,1,1];
for i = 1 : size(unitData,1)
    if isequal(responsive(i,:), target)
        num = num + 1;
    end
end
disp(num);

%% How many cells are aligned to the center
bin_size = 80;
bin_center_size = 2; % 2 bins around the onset of the event (= 50ms * 2 = 100ms around the event = 200ms)

data = first_LICK_zscores(logical(responsive(:,1)), :);
[~, peak_index] = max(data, [], 2);
[first_LICK_type_count, ~, first_LICK_type] = histcounts(peak_index, [1, bin_size/2-bin_center_size+1, bin_size/2+bin_center_size+1, 80]);
first_LICK_type_range = cumsum(first_LICK_type_count);
array2append = zeros(size(responsive,1),1);
array2append(logical(responsive(:,1))) = first_LICK_type;
output = [output, table(array2append, 'VariableNames', "first_LICK_type")];

data = valid_IROF_A_zscores(logical(responsive(:,2)), :);
[~, peak_index] = max(data, [], 2);
[valid_IROF_A_type_count, ~, valid_IROF_A_type] = histcounts(peak_index, [1, bin_size/2-bin_center_size+1, bin_size/2+bin_center_size+1, 80]);
valid_IROF_A_type_range = cumsum(valid_IROF_A_type_count);
array2append = zeros(size(responsive,1),1);
array2append(logical(responsive(:,2))) = valid_IROF_A_type;
output = [output, table(array2append, 'VariableNames', "valid_IROF_A_type")];

data = valid_IROF_E_zscores(logical(responsive(:,3)), :);
[~, peak_index] = max(data, [], 2);
[valid_IROF_E_type_count, ~, valid_IROF_E_type] = histcounts(peak_index, [1, bin_size/2-bin_center_size+1, bin_size/2+bin_center_size+1, 80]);
valid_IROF_E_type_range = cumsum(valid_IROF_E_type_range);
array2append = zeros(size(responsive,1),1);
array2append(logical(responsive(:,3))) = valid_IROF_E_type;
output = [output, table(array2append, 'VariableNames', "valid_IROF_E_type")];


%% Draw Peak Sorted PETH - Lick vs AHW vs EHW
figureSize = [89, 248, 288, 689];
figure('Name', 'SortedPETH_Lick', 'Position', figureSize);
ax_hm1 = subplot(4,1,1:3);
ax_hist1 = subplot(4,1,4);
drawPeakSortedPETH(first_LICK_zscores(logical(responsive(:,1)), :), [-2000, 2000], 50, ax_hm1, ax_hist1, 'Name', 'First Lick');
ax_hm1.Clipping = 'off';
hold(ax_hm1, 'on');
fill(ax_hm1, [82, 82, 85, 85], [1, first_LICK_type_range(1), first_LICK_type_range(1), 1], xkcd.red, 'LineStyle', 'None');
fill(ax_hm1, [82, 82, 85, 85], [first_LICK_type_range(1), first_LICK_type_range(2), first_LICK_type_range(2), first_LICK_type_range(1)], xkcd.goldenrod, 'LineStyle', 'None');
fill(ax_hm1, [82, 82, 85, 85], [first_LICK_type_range(2), first_LICK_type_range(3), first_LICK_type_range(3), first_LICK_type_range(2)], xkcd.blue, 'LineStyle', 'None');

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

%% Load Feature Importance Data
load('Output.mat');
r_ = [];
for i = 1 : 40
    r_ = [r_; mean(result{i}.importance_unit_score,1)'];
end
output = [output, table(r_, 'VariableNames', "FI")];

%% FI
mean(output.FI(output.first_LICK_type == 1))
mean(output.FI(output.first_LICK_type == 2))
mean(output.FI(output.first_LICK_type == 3))

mean(output.FI(output.valid_IROF_A_type == 1))
mean(output.FI(output.valid_IROF_A_type == 2))
mean(output.FI(output.valid_IROF_A_type == 3))

mean(output.FI(output.valid_IROF_E_type == 1))
mean(output.FI(output.valid_IROF_E_type == 2))
mean(output.FI(output.valid_IROF_E_type == 3))




