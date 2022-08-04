function drawSortedZvalue(binnedZ)
%% DrawSortedZvalue
% Draw Peak sorted PETH
% Input : m x n array. m : total cell number, n : total bin number. 
%       ex) 100 cells with 40 bins -> 100 x 40 matrix

TIMEWINDOW_LEFT = -1000; %(ms)
TIMEWINDOW_RIGHT = +1000; %(ms)
TIMEWINDOW_BIN = 50; %(ms) 
numBin = (TIMEWINDOW_RIGHT - TIMEWINDOW_LEFT)/TIMEWINDOW_BIN; % number of bins

[~, i] =max(binnedZ,[],2);
[~, ix] = sort(i);

new_result = binnedZ(ix,:);

figure('Position', [-1410, 112, 560, 828]);
ax1 = subplot(4,1,1:3);
imagesc(new_result);
hold on;
line([20.5,20.5],[1,size(binnedZ,1)], 'Color', 'w', 'LineWidth', 1); 
xticklabels({'-1000', '-500', '0', '+500', '+1000'});
xticks(0.5 : 10 : numBin + 0.5);
xlim([0.5,40.5]);
ylabel('Cell Number');
title('Avoidance Head Withdrawal (Responsive Only)');
colormap 'jet';
caxis([-5, 20]);
set(gca, 'FontName', 'Noto Sans');
colorbar;
drawnow;

% Peak Histogram
[~,i] = max(binnedZ,[],2);
ax2 = subplot(4,1,4);
histogram(i,(1:numBin)-0.5, 'FaceColor','k','LineStyle','none');
hold on;
ylim_ = ylim();
line([20,20],ylim_, 'Color', 'r', 'LineWidth', 1); 
xlim([0.5, 39.5]);
ylim(ylim_);
xticks([0.5, 10, 20, 30, 39.5]);
xticklabels({'-1000', '-500', '0', '+500', '+1000'});
ylabel('Number of Unit');
xlabel('Time (ms)');
title('Peak Histogram');
ax2.Position(3) = ax1.Position(3);
set(gca, 'FontName', 'Noto Sans');
end