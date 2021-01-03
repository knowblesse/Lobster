function ax = drawPETH(unit, TIMEWINDOW)
%% Draw PETH
% unit : cell array. each cell represent one trial and it has n x 1 matrix.
% n represent total spikes in that window.
% This function returns two axes in a cell structure. Setting the Parent
% property of each element can plot the axes to designated figure. 
%% Constants
HISTOGRAM_WIDTH = 50;

%% Raster Plot
numTrial = numel(unit);
ax1 = subplot(3,1,1:2);
for t = 1 : numel(unit)
    for s = 1 : numel(unit{t})
        line(ax1,[unit{t}(s),unit{t}(s)],[numTrial - t + 1, numTrial - t],'Color','k');
    end
end
line(ax1,[0,0],[0,numel(unit)],'Color','r','LineWidth',1);
ylabel('Trial');
xlim(TIMEWINDOW);
ylim([0,numel(unit)]);
yticklabels(numTrial - yticks);

%% Histogram
ax2 = subplot(3,1,3);
unit_all = cat(1,unit{:});
[N,edges] = histcounts(unit_all,TIMEWINDOW(1):HISTOGRAM_WIDTH:TIMEWINDOW(2));
bar(ax2,edges(1:end-1) + HISTOGRAM_WIDTH / 2,N,'FaceColor','k','LineStyle','none','BarWidth',1);
xlim(TIMEWINDOW);
ylabel('Cell Count');
xticks([]);
xlabel('Time(ms)');

%% Return
ax = {ax1,ax2};
end
