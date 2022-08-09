%% PlotRepresentativeBehaviorData


%% Version1
TANK_location = 'F:\LobsterData\#21JAN5-210716-170133_PL';
dat = BehavDataParser(TANK_location);
figure('Position', [-1558, 655, 990, 254]);
plot([]);
hold on;
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
ylim([0, 3]);
yticks([0.5, 1.5, 2.5]);
yticklabels(["Trial 3", "Trial 2", "Trial 1"]);
xticks(1:10);
xlabel('Time (sec)');
set(gca, 'FontName', 'Noto Sans');

%% Version 2
TANK_location = 'F:\LobsterData\#21JAN5-210716-170133_PL';
dat = BehavDataParser(TANK_location);
figure('Position', [-1558, 655, 550, 130]);
plot([]);
hold on;
startTrial = 12;
numTrial = 2;

for trial = startTrial : startTrial + numTrial -1
    
    ttime = dat{trial,1}(1); % start time
    % Draw TRON / TROF
    line([dat{trial,1}(1), dat{trial,1}(1)], [-1, 1], 'Color', 'k', 'LineWidth', 2);
    text(dat{trial,1}(1), 1.5, 'Trial Start', 'HorizontalAlignment', 'center', 'FontName', 'Noto Sans', 'FontSize', 6);
    line([dat{trial,1}(2), dat{trial,1}(2)], [-1, 1], 'Color', 'k', 'LineWidth', 2);
    text(dat{trial,1}(2), 1.5, 'Trial End', 'HorizontalAlignment', 'center', 'FontName', 'Noto Sans', 'FontSize', 6);
    
    % Draw IRs
    for irs = 1 : size(dat{trial, 2}, 1)
        fill([dat{trial, 2}(irs,1), dat{trial, 2}(irs,2), dat{trial, 2}(irs,2), dat{trial, 2}(irs,1)] + ttime, [-1, -1, 1, 1], [0, 185, 227] ./255 , 'FaceAlpha', 0.4, 'LineStyle', 'none');
    end
    
    % Draw Licks
    for lks = 1 : size(dat{trial, 3}, 1)
        line([dat{trial, 3}(lks), dat{trial, 3}(lks)] + ttime, [-1,1], 'Color', [56,212,48]./255, 'LineWidth',1);
    end
    
    % Draw Attk
    line([dat{trial, 4}(1), dat{trial, 4}(1)] + ttime, [-1,1], 'Color', 'r', 'LineWidth',2);
    text(dat{trial,4}(1) + ttime, 1.5, 'Attack', 'HorizontalAlignment', 'center', 'FontName', 'Noto Sans', 'FontSize', 6);
end

% Mark Timer

fill([dat{startTrial, 3}(1), dat{startTrial, 4}(1), dat{startTrial, 4}(1), dat{startTrial, 3}(1)] + dat{startTrial,1}(1), [-1.5, -1.5, -1.3, -1.3], 'k' , 'FaceAlpha', 0.2, 'LineStyle', 'none');
text(mean([dat{startTrial, 3}(1), dat{startTrial, 4}(1)] + dat{startTrial,1}(1)), -1.4, '6 sec', 'Color', 'k', 'HorizontalAlignment', 'center', 'FontName', 'Noto Sans', 'FontSize', 6);
fill([dat{startTrial+1, 3}(1), dat{startTrial+1, 4}(1), dat{startTrial+1, 4}(1), dat{startTrial+1, 3}(1)] + dat{startTrial+1,1}(1), [-1.5, -1.5, -1.3, -1.3], 'k' , 'FaceAlpha', 0.2, 'LineStyle', 'none');
text(mean([dat{startTrial+1, 3}(1), dat{startTrial+1, 4}(1)] + dat{startTrial+1,1}(1)), -1.4, '3 sec', 'Color', 'k', 'HorizontalAlignment', 'center', 'FontName', 'Noto Sans', 'FontSize', 6);

xlabel('Time (sec)');
xlim([152, 174]);
xticks(152 : 2 : 174);
ylim([-2, 2]);
yticks([]);

set(gca, 'FontName', 'Noto Sans');

