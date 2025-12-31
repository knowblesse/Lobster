%% ConcatAllBehaviors
% Concat all behaviors (HW and Licks) in all sessions

basePath = 'E:\Data\Lobster\BehaviorData';

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

interLickinterval = [];
maxInterLickinterval = [];

withinTrialLickInterval = [];
withinSessionLickInterval = [];
%% Session
for session = 1 : 49
    TANK_name = cell2mat(sessionPaths{session});
    TANK_location = char(strcat(basePath, filesep, TANK_name));
    % Scripts
    load(fullfile(basePath, TANK_name));
    
    numTrial = size(ParsedData,1);

    lickCounts_6_ = [];
    lickCounts_3_ = [];
    
    trialLickIntervals = [];
    for trial = 1 : numTrial 
        numLick = size(ParsedData{trial,3},1);
        interLickinterval = [interLickinterval;...
            ParsedData{trial,3}(2:end,2) - ParsedData{trial,3}(1:end-1,1)];
        maxInterLickinterval = [maxInterLickinterval;...
            max(ParsedData{trial,3}(2:end,2) - ParsedData{trial,3}(1:end-1,1))];
        % Within Session 
        
        % Calc first half and last half of interLickinterval
        % First, check if the num lick is more than 4
        if  numLick < 4
            continue;
        else
            trialLickIntervals = [trialLickIntervals; mean(ParsedData{trial,3}(2:end,2) - ParsedData{trial,3}(1:end-1,1))];
            halfPoint = round(numLick/2);
            firstHalf = mean(ParsedData{trial,3}(2:halfPoint,2) - ParsedData{trial,3}(1:halfPoint-1,1));
            lastHalf = mean(ParsedData{trial,3}(halfPoint+1:end,2) - ParsedData{trial,3}(halfPoint:end-1,1));
            withinTrialLickInterval = [withinTrialLickInterval; lastHalf - firstHalf];
        end
    end
    withinSessionLickInterval = [withinSessionLickInterval;...
        [...
            mean(trialLickIntervals(round(numel(trialLickIntervals)/2) : end) - ...
            mean(trialLickIntervals(1:round(numel(trialLickIntervals)/2)))...
            )
        ]];
end
fprintf('DONE\n');
%%
figure('Position', [2213, 604, 1165, 379]);
subplot(1,3,1);
histogram(interLickinterval, 0:0.01:2, 'FaceColor', 'black', 'LineStyle', 'None')
val = ylim;
line([median(interLickinterval), median(interLickinterval)], [val(1), val(2)], 'Color', 'r');
ylabel('count')
xlabel('Inter Lick Interval (s)')
title('Inter-Lick-Intervals');

subplot(1,3,2);
histogram(withinTrialLickInterval, -0.5:0.01:0.5, 'FaceColor', 'black', 'LineStyle', 'None')
val = ylim;
line([median(withinTrialLickInterval), median(withinTrialLickInterval)], [val(1), val(2)], 'Color', 'r');
ylabel('count')
xlabel('ILI Difference (s)')
title('Within-Trial');

subplot(1,3,3);
histogram(withinSessionLickInterval, -0.1:0.02:0.1, 'FaceColor', 'black', 'LineStyle', 'None')
val = ylim;
line([median(withinSessionLickInterval), median(withinSessionLickInterval)], [val(1), val(2)], 'Color', 'r');
ylabel('count')
xlabel('ILI Difference (s)')
title('Within-Session');

median(interLickinterval)
%%
histogram(maxInterLickinterval, 0:0.01:2, 'FaceColor', 'black', 'LineStyle', 'None')
ylabel('count')
xlabel('max(Inter Lick Interval) (s)')
median(maxInterLickinterval)

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