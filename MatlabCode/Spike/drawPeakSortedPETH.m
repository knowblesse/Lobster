function sortOrder = drawPeakSortedPETH(zscores, TIMEWINDOW, TIMEWINDOW_BIN, ax_hm, ax_hist, options)
%% drawPeakSortedPETH
arguments
    zscores (:,:) double % vertical vector is one data. sorted horizontally.
    TIMEWINDOW (1,2) double 
    TIMEWINDOW_BIN (1,1) double
    ax_hm matlab.graphics.axis.Axes
    ax_hist matlab.graphics.axis.Axes
    options.ManualIndex = []
    options.Name char = ''
    options.Sort = true
end

numBin = diff(TIMEWINDOW)/TIMEWINDOW_BIN; % number of bins

if options.Sort
    if isempty(options.ManualIndex)
        [~, peak_index] = max(zscores, [], 2);
        [~, sortOrder] = sort(peak_index);
        sorted_zscores = zscores(sortOrder, :);
    else
        sorted_zscores = zscores(options.ManualIndex, :);
    end
else
    sorted_zscores = zscores;
end

%% Heat Map
imagesc(ax_hm, sorted_zscores);
hold on;
line(ax_hm, [numBin, numBin]/2+0.5,[1,size(sorted_zscores,1)], 'Color', 'w', 'LineWidth', 0.8); 
xticks(ax_hm, 0.5 : 20 : numBin + 0.5);
xticklabels(ax_hm, arrayfun(@num2str, (TIMEWINDOW(1):1000:TIMEWINDOW(2))/1000, 'UniformOutput', false))
xlim(ax_hm, [0.5, numBin + 0.5]);
%ylabel(ax_hm, 'Unit');
if ~isempty(options.Name)
    title(ax_hm, options.Name);
end
colormap 'jet';
caxis(ax_hm, [-5, 20]);
set(ax_hm, 'FontName', 'Noto Sans');
drawnow;

% This function tells where the peak of each unit's firing rate is located on the time axis.
% %% Peak Histogram
% histogram(ax_hist, peak_index,(1:numBin)-0.5, 'Normalization', 'pdf', 'FaceColor','k','LineStyle','none');
% hold on;
% ylim_ = ylim;
% %ylim_ = [0, 0.1];
% line(ax_hist, [numBin, numBin]/2, ylim_, 'Color', 'r', 'LineWidth', 0.8); 
% xlim(ax_hist, [0.5, numBin - 0.5]);
% %ylim(ax_hist, ylim_);
% xticks(ax_hist, [0.5, 10 : 10 : numBin-10, numBin - 0.5]);
% xticklabels(ax_hist, arrayfun(@num2str, TIMEWINDOW(1):500:TIMEWINDOW(2), 'UniformOutput', false))
% %ylabel(ax_hist, 'Number of Unit');
% xlabel(ax_hist, 'Time (ms)');
% ax_hist.Position(3) = ax_hm.Position(3);
% set(ax_hist, 'FontName', 'Noto Sans');

%% Mean Z score
set(ax_hist, 'FontName', 'Noto Sans');
plot(mean(clip(sorted_zscores, -5, 5), 1), 'Color','k', 'LineWidth',1);
hold on;
ylim_ = ylim;
line(ax_hist, ones(1,2) * size(sorted_zscores,2) / 2, [-5, 5], 'Color', 'r', 'LineWidth', 0.8); 
line(ax_hist, [0.5, numBin + 0.5], [0,0], 'Color', 'k', 'LineWidth', 0.8, 'LineStyle', '--'); 
xlim(ax_hist, [0.5, numBin + 0.5]);
xticks(ax_hist, 0.5 : 20 : numBin + 0.5);
xticklabels(ax_hist, arrayfun(@num2str, (TIMEWINDOW(1):1000:TIMEWINDOW(2))/1000, 'UniformOutput', false))
ylabel(ax_hist, 'Z');
xlabel(ax_hist, 'Time (ms)');
ax_hist.Position(3) = ax_hm.Position(3);


end

