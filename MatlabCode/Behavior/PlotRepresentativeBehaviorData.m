%% PlotRepresentativeBehaviorData

TANK_location = 'F:\LobsterData\#21JAN5-210716-170133_PL';
dat = BehavDataParser(TANK_location);

figure('Position', [-1558, 655, 990, 254]);
plot([]);
hold on;
%% Trial
startTrial = 12;
numTrial = 3;
for trial = startTrial : startTrial + numTrial -1
    line([diff(dat{trial, 1}), diff(dat{trial, 1})] , [numTrial-1,numTrial], 'Color', 'k', 'LineWidth', 2);
    for irs = 1 : size(dat{trial, 2}, 1)
        fill([dat{trial, 2}(irs,1), dat{trial, 2}(irs,2), dat{trial, 2}(irs,2), dat{trial, 2}(irs,1)], [numTrial - 1, numTrial - 1, numTrial, numTrial], 'b', 'FaceAlpha', 0.1, 'LineStyle', 'none');
    end
    
    for lks = 1 : size(dat{trial, 3}, 1)
        line([dat{trial, 3}(lks), dat{trial, 3}(lks)] , [numTrial-1,numTrial], 'Color', [56,212,48]./255, 'LineWidth',2);
    end
    
    line([dat{trial, 4}(1), dat{trial, 4}(1)] , [numTrial-1,numTrial], 'Color', 'r', 'LineWidth',2);
    
    numTrial = numTrial - 1;
end

yticks([0.5, 1.5, 2.5]);
yticklabels(["Trial 3", "Trial 2", "Trial 1"]);
xticks(1:10);
xlabel('Time (sec)');
set(gca, 'FontName', 'Noto Sans');
