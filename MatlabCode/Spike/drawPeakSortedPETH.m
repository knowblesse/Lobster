function drawPeakSortedPETH(zscores, TIMEWINDOW, TIMEWINDOW_BIN, ax_hm, ax_hist, options)
%% drawPeakSortedPETH
arguments
    zscores (:,:) double % vertical vector is one data. sorted horizontally.
    TIMEWINDOW (1,2) double 
    TIMEWINDOW_BIN (1,1) double
    ax_hm matlab.graphics.axis.Axes
    ax_hist matlab.graphics.axis.Axes
    options.ManualIndex = []
    options.Name char = ''
end

numBin = diff(TIMEWINDOW)/TIMEWINDOW_BIN; % number of bins

if isempty(options.ManualIndex)
    [~, peak_index] = max(zscores, [], 2);
    [~, ix] = sort(peak_index);
    sorted_zscores = zscores(ix, :);
else
    sorted_zscores = zscores(options.ManualIndex)
end

%% Heat Map
imagesc(ax_hm, sorted_zscores);
hold on;
line(ax_hm, [numBin, numBin]/2+0.5,[1,size(sorted_zscores,1)], 'Color', 'w', 'LineWidth', 1); 
xticks(ax_hm, 0.5 : 20 : numBin + 0.5);
xticklabels(ax_hm, arrayfun(@num2str, TIMEWINDOW(1):1000:TIMEWINDOW(2), 'UniformOutput', false))
xlim(ax_hm, [0.5, numBin + 0.5]);
%ylabel(ax_hm, 'Unit');
if ~isempty(options.Name)
    title(ax_hm, options.Name);
end
colormap 'jet';
caxis(ax_hm, [-5, 20]);
set(ax_hm, 'FontName', 'Noto Sans');
drawnow;

%% Peak Histogram
histogram(ax_hist, peak_index,(1:numBin)-0.5, 'Normalization', 'pdf', 'FaceColor','k','LineStyle','none');
hold on;
ylim_ = ylim;
%ylim_ = [0, 0.1];
line(ax_hist, [numBin, numBin]/2, ylim_, 'Color', 'r', 'LineWidth', 1); 
xlim(ax_hist, [0.5, numBin - 0.5]);
%ylim(ax_hist, ylim_);
xticks(ax_hist, [0.5, 10 : 10 : numBin-10, numBin - 0.5]);
xticklabels(ax_hist, arrayfun(@num2str, TIMEWINDOW(1):500:TIMEWINDOW(2), 'UniformOutput', false))
%ylabel(ax_hist, 'Number of Unit');
xlabel(ax_hist, 'Time (ms)');
ax_hist.Position(3) = ax_hm.Position(3);
set(ax_hist, 'FontName', 'Noto Sans');

end
