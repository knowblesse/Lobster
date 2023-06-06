%% ConcatAllBehaviors
% Concat all behaviors (HW and Licks) in all sessions

basePath = 'D:\Data\Lobster\BehaviorData';

filelist = dir(basePath);
sessionPaths = regexp({filelist.name},'^#\S*','match');
sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));
fprintf('%d sessions detected.\n', numel(sessionPaths));

%% Outputs
allHWTimes_6 = []; % all last HW time (valid HW) during 6 sec attack trials
allHWTimes_3 = [];

allLickCounts_6 = []; % total number of licks per trial during 6 sec attack trials
allLickCounts_3 = [];

lickCounts_6 = cell(40,1); % total licks per trial. sustain trial number
lickCounts_3 = cell(40,1);

%% Session
for session = 1 : 40
    TANK_name = cell2mat(sessionPaths{session});
    TANK_location = char(strcat(basePath, filesep, TANK_name));
    % Scripts
    load(fullfile(basePath, TANK_name));
    
    numTrial = size(ParsedData,1);

    lickCounts_6_ = [];
    lickCounts_3_ = [];
    for trial = 1 : numTrial
        % Find valid IROF
        nearAttackIRindex = find(ParsedData{trial,2}(:,1) < ParsedData{trial,4}(1), 1, 'last');
        HWTime = ParsedData{trial,2}(nearAttackIRindex,2) - ParsedData{trial,3}(1);
        numLick = size(ParsedData{trial,3},1);
        if isAttackIn3Sec(trial) == 1
            allHWTimes_3 = [allHWTimes_3; HWTime];
            allLickCounts_3 = [allLickCounts_3; numLick];
            lickCounts_3_ = [lickCounts_3_; numLick];
        else
            allHWTimes_6 = [allHWTimes_6; HWTime];
            allLickCounts_6 = [allLickCounts_6; numLick];
            lickCounts_6_ = [lickCounts_6_; numLick];
        end
    end
    lickCounts_6{session} = lickCounts_6_;
    lickCounts_3{session} = lickCounts_3_;
end
fprintf('DONE\n');
clearvars lickCounts_3_ lickCounts_6_

%% Draw Histograms - HW (6 sec)
figure('Position', [0, 500, 265, 218]);
clf;
histogram(allHWTimes_6, 0:0.2:6.2, 'FaceColor', 'k', 'LineStyle','none');
xlim([0, 6.2]);
ylim([0, 200]);
hold on;
line([6, 6], [0, 200], 'Color', 'r');
ylabel('Count');
xlabel('HW Time (sec)');
set(gca, 'FontName', 'Noto Sans');
set(gca, 'FontSize', 7);

%% Draw Histogram - HW (3 sec)
figure('Position', [0, 500, round(265/2), 218]);
clf;
histogram(allHWTimes_3, 0:0.2:3.2, 'FaceColor', 'k', 'LineStyle','none');
xlim([0, 3.2]);
ylim([0, 100]);
hold on;
line([3, 3], [0, 100], 'Color', 'r');
xticks(0:3);
ylabel('Count');
xlabel('HW Time (sec)');
set(gca, 'FontName', 'Noto Sans');
set(gca, 'FontSize', 7);

%% Draw Histograms - Lick (6 sec)
figure('Position', [0, 300, 265, 218]);
clf;
histogram(allLickCounts_6, 0:2:100, 'FaceColor', 'k', 'LineStyle','none');
xlim([0, 100]);
ylim([0, 120]);
ylabel('Count');
xlabel('Lick count per trial');
set(gca, 'FontName', 'Noto Sans');
set(gca, 'FontSize', 7);

%% Draw Histogram - Lick (3 sec)
figure('Position', [0, 300, round(265/2), 218]);
clf;
histogram(allLickCounts_3, 0:2:60, 'FaceColor', 'k', 'LineStyle','none');
xlim([0, 60]);
ylim([0, 100]);
ylabel('Count');
xlabel('Lick count per trial');
xticks(0:20:60);
set(gca, 'FontName', 'Noto Sans');
set(gca, 'FontSize', 7);

%% Lick Count Transition (6 sec)
figure('Position', [0, 300, 265, 218]);
clf;

stretchedLickCount = zeros(40,100);
for session = 1 : 40
    numTrial = size(lickCounts_6{session},1);
    stretchedLickCount(session, :) = interp1(1:numTrial, lickCounts_6{session}, linspace(1, numTrial, 100));
end

shadeplot(stretchedLickCount, 'SD', 'sem', 'Color', 'k', 'LineWidth', 1)
xlim([0, 100]);
ylim([0, 60]);
ylabel('Lick Count');
xlabel('Trial progression (%)');
xticks(0:20:100);
set(gca, 'FontName', 'Noto Sans');
set(gca, 'FontSize', 7);

%% Lick Count Transition (3 sec)
figure('Position', [0, 300, round(265/2), 218]);
clf;

stretchedLickCount = zeros(40,100);
for session = 1 : 40
    numTrial = size(lickCounts_3{session},1);
    stretchedLickCount(session, :) = interp1(1:numTrial, lickCounts_3{session}, linspace(1, numTrial, 100));
end

shadeplot(stretchedLickCount, 'SD', 'sem', 'Color', 'k', 'LineWidth', 1)
xlim([0, 100]);
ylim([0, 60]);
ylabel('Lick Count');
xlabel('Trial progression (%)');
xticks(0:20:100);
set(gca, 'FontName', 'Noto Sans');
set(gca, 'FontSize', 7);