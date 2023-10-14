%% AnalyzeFineDistanceData

basePath = 'D:\Data\Lobster\FineDistanceResult_syncFixed_June';
behavDataPath = 'D:\Data\Lobster\BehaviorData';

filelist = dir(basePath);
sessionPaths = regexp({filelist.name},'^#\S*.mat','match');
sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));

load("Apparatus.mat");
px2cm = 0.169;
truncatedTimes_s = 10;
neural_data_rate = 20;

%% Load Behavior Data
numSession = numel(sessionPaths);
data_behav = cell(1,numSession);
for session = 1 : numSession
    TANK_name = cell2mat(sessionPaths{session});
    TANK_location = char(strcat(basePath, filesep, TANK_name));
    
    temp = regexp(TANK_name, '(?<tname>#.*?)result.*', 'names');

    load(fullfile(behavDataPath, strcat(temp.tname, '.mat')));
    data_behav{session} = ParsedData;
end

%% Load Data by session
data = cell(1,numSession);
midPointTimesData = cell(1,numSession);
numCell = zeros(numSession, 1);
for session = 1 : numSession
    TANK_name = cell2mat(sessionPaths{session});
    TANK_location = char(strcat(basePath, filesep, TANK_name));
    load(TANK_location, "WholeTestResult", "midPointTimes"); % PFITestResult, WholeTestResult(row, col, true d , shuffled d, pred d)
    data{session} = WholeTestResult;
    if exist("midPointTimes") == 0
            midPointTimesData{session} = truncatedTimes_s + (1/neural_data_rate)*(0:size(WholeTestResult,1)-1) + 0.5 * (1/neural_data_rate);
    else
        midPointTimesData{session} = midPointTimes;
    end
end

%% Compare Error btw shuffled and predicted
result1 = table(zeros(numSession,1), zeros(numSession,1), 'VariableNames',["Shuffled", "Predicted"]);
for session = 1 : numSession
    if contains(basePath, 'degree')
        % if degree regressor, first get abs difference, and check if
        % 360-diff is smaller than diff.
        result1.Shuffled(session) = mean(min([...
            abs(data{session}(:,3) - data{session}(:,4)),...
            360 - (abs(data{session}(:,3) - data{session}(:,4)))...
            ], [], 2));
        result1.Predicted(session) = mean(min([...
            abs(data{session}(:,3) - data{session}(:,5)),...
            360 - (abs(data{session}(:,3) - data{session}(:,5)))...
            ], [], 2));
    elseif contains(basePath, 'time')
        result1.Shuffled(session) = mean(abs(data{session}(:,3) - data{session}(:,4))) / 60;
        result1.Predicted(session) = mean(abs(data{session}(:,3) - data{session}(:,5))) / 60;
    else
        result1.Shuffled(session) = mean(abs(data{session}(:,3) - data{session}(:,4))) * px2cm;
        result1.Predicted(session) = mean(abs(data{session}(:,3) - data{session}(:,5))) * px2cm;
    end
end

%% Compare Error btw N, F, and E
result2 = table(zeros(numSession,1), zeros(numSession,1), zeros(numSession,1), 'VariableNames', ["NestError", "ForagingError", "EncounterError"]);
result2_normalized = table(zeros(numSession,1), zeros(numSession,1), zeros(numSession,1), 'VariableNames', ["NestError", "ForagingError", "EncounterError"]);
datapointNumber = table(zeros(numSession,1), zeros(numSession,1), zeros(numSession,1), 'VariableNames', ["Nest", "Foraging", "Encounter"]);
for session = 1 : numSession
    locError = abs(data{session}(:,3) - data{session}(:,5));
    
    isNesting = data{session}(:,2) < 225;
    
    % check isEncounter by IRsensor
    IRs = [];
    for trial = 1 : size(data_behav{session},1)
       IRs = [IRs; data_behav{session}{trial,2} + data_behav{session}{trial,1}(1)];
    end
    
    isEncounter = false(size(midPointTimesData{session}, 2),1);
    for i = 1 : size(isEncounter,1)
       isEncounter(i) = any(IRs(:,1) < midPointTimesData{session}(i) & midPointTimesData{session}(i) < IRs(:,2));
    end
    
    isForaging = and(~isNesting, ~isEncounter);

    fprintf("%2d session : N-%d, F-%d, E-%d\n", session, sum(isNesting), sum(isForaging), sum(isEncounter));

    result2.NestError(session) = mean(locError(isNesting)) * px2cm;
    result2.ForagingError(session) = mean(locError(isForaging)) * px2cm;
    result2.EncounterError(session) = mean(locError(isEncounter)) * px2cm;
    
    
    % Normalized error
    std_nest = std(data{session}(isNesting,3));
    std_foraging = std(data{session}(isForaging,3));
    std_encounter = std(data{session}(isEncounter,3));
    
    result2_normalized.NestError(session) = mean(locError(isNesting)/std_nest) * px2cm;
    result2_normalized.ForagingError(session) = mean(locError(isForaging)/std_foraging) * px2cm;
    result2_normalized.EncounterError(session) = mean(locError(isEncounter)/std_encounter) * px2cm;
    
    % Datapoint
    datapointNumber.Nest(session) = sum(isNesting);
    datapointNumber.Foraging(session) = sum(and(~isNesting, ~isEncounter));
    datapointNumber.Encounter(session) = sum(isEncounter);
end

%% Separate Forgaging zone error by outbound and inbound path
result3 = table(zeros(numSession,1), zeros(numSession,1), 'VariableNames', ["OutboundError", "InboundError"]);

for session = 1 : numSession
    locError = abs(data{session}(:,3) - data{session}(:,5));
    
    isNesting = data{session}(:,2) < 225;
    
    % check isEncounter by IRsensor
    IRs = [];
    for trial = 1 : size(data_behav{session},1)
       IRs = [IRs; data_behav{session}{trial,2} + data_behav{session}{trial,1}(1)];
    end
    
    isEncounter = false(size(midPointTimesData{session}, 2),1);
    for i = 1 : size(isEncounter,1)
       isEncounter(i) = any(IRs(:,1) < midPointTimesData{session}(i) & midPointTimesData{session}(i) < IRs(:,2));
    end
    
    isForaging = and(~isNesting, ~isEncounter);

    % get outbound index : (from door open to first enterance to E-zone) & isForaging
    isOutbound = false(size(midPointTimesData{session}, 2),1);
    for trial = 2 : size(data_behav{session},1) % ignore the first trial, because rat start from weird location
        tron = data_behav{session}{trial, 1}(1);
        isOutbound = isOutbound | (...
            midPointTimesData{session}' >= tron & ...
            midPointTimesData{session}' < (tron + data_behav{session}{trial, 3}(1)));
    end

    % get inbound index : (from last ir to next trial's door open) & isForaging
    isInbound = false(size(midPointTimesData{session}, 2),1);
    for trial = 1 : size(data_behav{session},1)-1 % ignore the last trial, because we use the next trial's TRON
        tron = data_behav{session}{trial, 1}(1);
        isInbound = isInbound | (...
            midPointTimesData{session}' >= (tron + data_behav{session}{trial, 2}(end)) & ...
            midPointTimesData{session}' < data_behav{session}{trial+1, 1}(1));
    end
    
    result3.OutboundError(session) = mean(locError(isForaging & isOutbound)) * px2cm;
    result3.InboundError(session) = mean(locError(isForaging & isInbound)) * px2cm;
    
    fprintf("%2d session\n", session);
end

%% Error vs Time in session
result4 = zeros(numSession,10000);

for session = 1 : numSession
    locError = abs(data{session}(:,3) - data{session}(:,5));
    
    numDataPoint = size(locError,1);
    
    result4(session,:) = interp1(1:numDataPoint, locError .* px2cm, linspace(1, numDataPoint, 10000));
    
    fprintf("%2d session\n", session);
end

%% Draw time vs error plot
figure('Position', [0,300, 400, 300]);
clf;
shadeplot(result4, 'SD', 'sd', 'Color', 'k', 'LineWidth', 1);
hold on;
X = [ones(10000,1), (1:10000)'];
Y = mean(result4,1)';
b1 = X\Y;

plot(X * b1, 'Color', 'r', 'LineWidth', 2);
xlabel('Session progression (%)');
xticks(0:2000:10000);
xticklabels(0:20:100);
ylabel('Mean absolute error (cm)');
set(gca, 'FontName', 'Noto Sans', 'FontSize', 8)
mdl = fitlm(x, Y);

%%  Draw offset plot
session = 20;
figure('Position', [0,300, 400, 300]);
[val, x] = xcorr(data{session}(:,3), data{session}(:,5), 10 / 0.050);
plot(x * 0.050, val, 'Color', 'k', 'LineWidth', 1);
hold on;
yrange = ylim;
line([0, 0], [yrange(1), yrange(2)], 'Color', 'r', 'LineStyle', '--');
ylim(yrange);
xlabel('Offset (sec)');
ylabel('cross-correlation (a.u.)');
set(gca, 'FontName', 'Noto Sans', 'FontSize', 8);

%% Draw TRON and HE's true and predicted distance
session = 7;

numTrial = size(data_behav{session},1);
TRON_time_data = zeros(80, numTrial-2, 2); % 2sec before, 2sec after => 50ms x 80;
LICK_time_data = zeros(80, numTrial-2, 2); % 2sec before, 2sec after => 50ms x 80;
for trial = 2 : numTrial-1
    TRON_time = data_behav{session}{trial,1}(1);
    TRON_index = find(midPointTimesData{session}>TRON_time,1);
    TRON_time_data(:, trial-1, 1) = data{session}(TRON_index-40:TRON_index+39, 3);
    TRON_time_data(:, trial-1, 2) = data{session}(TRON_index-40:TRON_index+39, 5);

    LICK_time = TRON_time + data_behav{session}{trial,3}(1);
    LICK_index = find(midPointTimesData{session}>LICK_time,1);
    LICK_time_data(:, trial-1, 1) = data{session}(LICK_index-40:LICK_index+39, 3);
    LICK_time_data(:, trial-1, 2) = data{session}(LICK_index-40:LICK_index+39, 5);
end

figure('Name', 'TRON plot', 'Position', [0, 300, 400, 300]);
clf;
[~,ax1]=shadeplot(TRON_time_data(:,:,1)'*px2cm, 'SD', 'sem', 'Color', 'k', 'LineStyle', '-', 'LineWidth', 1);
hold on;
[~,ax2]=shadeplot(TRON_time_data(:,:,2)'*px2cm, 'SD', 'sem', 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1);
yrange = ylim;
line([40,40], [yrange(1), yrange(2)], 'Color', 'r', 'LineStyle', '--')
xticks(0:20:80)
xticklabels(-2:2)
xlabel('Time from trial start (sec)');
ylabel('Distance (cm)');
legend([ax1, ax2], {'True', 'Predicted'});
set(gca, 'FontName', 'Noto Sans', 'FontSize', 8);

figure('Name', 'LICK plot', 'Position', [0, 300, 400, 300]);
clf;
[~,ax1]=shadeplot(LICK_time_data(:,:,1)'*px2cm, 'SD', 'sem', 'Color', 'k', 'LineStyle', '-', 'LineWidth', 1);
hold on;
[~,ax2]=shadeplot(LICK_time_data(:,:,2)'*px2cm, 'SD', 'sem', 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1);
yrange = ylim;
line([40,40], [yrange(1), yrange(2)], 'Color', 'r', 'LineStyle', '--')
xticks(0:20:80)
xticklabels(-2:2)
xlabel('Time from first lick (sec)');
ylabel('Distance (cm)');
legend([ax1, ax2], {'True', 'Predicted'});
set(gca, 'FontName', 'Noto Sans', 'FontSize', 8);


%% Draw Error Heatmap
% Apparatus Image Size
accumErrorMatrix = zeros(apparatus.height, apparatus.width);
accumLocationMatrix = zeros(apparatus.height, apparatus.width);

% Run through all sessions
for session = 1 : numSession
    locError = abs(data{session}(:,3) - data{session}(:,5)) * px2cm;
    for i = 1 : numel(locError)
        accumErrorMatrix(round(data{session}(i,1)), round(data{session}(i,2))) = ...
            accumErrorMatrix(round(data{session}(i,1)), round(data{session}(i,2))) + locError(i);
        
        accumLocationMatrix(round(data{session}(i,1)), round(data{session}(i,2))) = ...
            accumLocationMatrix(round(data{session}(i,1)), round(data{session}(i,2))) + 1;
    end
end

meanErrorMatrix = accumErrorMatrix ./ accumLocationMatrix;

% Location Index
locationMatrix = imgaussfilt(accumLocationMatrix, 20, 'FilterSize', 1001);
locationMatrix = locationMatrix .* apparatus.mask;

figure(1);
clf;
surf(accumLocationMatrix, 'LineStyle', 'none');
title('Number of visit in each pixel');

figure(2);
clf;
imshow(apparatus.image);
hold on;
colormap jet;
imagesc(locationMatrix, 'AlphaData', 0.5*ones(apparatus.height, apparatus.width));
contour(locationMatrix, 30, 'LineWidth',1.8);
title('Proportion of location');
caxis([0, 100]);
colorbar;
set(gca, 'FontName', 'Noto Sans');
set(gcf, 'Position', [428, 234, 731, 492]);

% Method 1 : Draw Normalized Error (Just for the reference, use the method 2 instead)
errorMatrix = imgaussfilt(accumErrorMatrix, 20, 'FilterSize', 1001);
normalizedErrorMatrix = errorMatrix ./ locationMatrix;
normalizedErrorMatrix(isnan(normalizedErrorMatrix(:))) = 0;

normalizedErrorMatrix = normalizedErrorMatrix .* apparatus.mask;

figure(3);
clf;
imshow(apparatus.image);
hold on;
colormap jet;
imagesc(normalizedErrorMatrix, 'AlphaData', 0.5*ones(apparatus.height, apparatus.width));
contour(normalizedErrorMatrix, 25, 'LineWidth',3);
title('Norm Error');

% Method 2 : Interpolate error for unknown location
%meanErrorMatrix = meanErrorMatrix .* apparatus.mask;
apparatus.mask(100:130, :) = 0;
x = [];
y = [];
v = [];
for row = 1 : 480
    for col = 1 : 640
        if ~isnan(meanErrorMatrix(row, col))
            if apparatus.mask(row, col) == 1
                x = [x, col];
                y = [y, row]; 
                v = [v, meanErrorMatrix(row, col)];
            end
        end
    end
end

[xq, yq] = meshgrid(1:640, 1:480);
f = scatteredInterpolant(x', y', v', 'natural', 'nearest');
vq = f(xq, yq);

figure(4);
clf;
scatter(x, y, 10, v, 'filled')
caxis([0, 400]);
colormap 'jet';
colorbar;
caxis([0, 400]);
xlim([40, 580])
ylim([120, 450]);
title('Raw error value matrix');

figure(5);
imagesc(vq);
title('Original Mean Distance L1 Error');
colormap 'jet'
colorbar
caxis([0, 400]);

% This is main figure
figure(6);
clf;
vq(isnan(vq)) = 0;
imshow(apparatus.image);
hold on;
colormap 'jet'
smoothedError = imgaussfilt(vq, 15, 'FilterSize', 1001) .* apparatus.mask;
imagesc(smoothedError, 'AlphaData', 0.3*(ones(480, 640)));
contour(smoothedError, 25, 'LineWidth', 1.8);
colorbar
caxis([0,30])
title('Smoothed Mean Distance L1 Error');
set(gca, 'FontName', 'Noto Sans');
set(gcf, 'Position', [428, 234, 731, 492]);

%% Correlation between smoothed Error vs locationMatrix
% to test whether the location with more data has low decoding accuracy
% the result shows that there is no correlation, indicating that the heterogenic accuracy
% distribution is not due to the imbalanced dataset.
[R, p] = corrcoef(locationMatrix(locationMatrix ~= 0 ), smoothedError(locationMatrix~=0))

%% Decoding error between Avoidance and Escape point
timewindow = [-10, 10];
numDatapoints = diff(timewindow) * neural_data_rate;

decodingError.HE.A = zeros(0,numDatapoints);
decodingError.HE.E = zeros(0,numDatapoints);
decodingError.HW.A = zeros(0,numDatapoints);
decodingError.HW.E = zeros(0,numDatapoints);

for session = 1 : 40
    numTrial = size(data_behav{session}, 1);
    behaviorResult = analyticValueExtractor(data_behav{session}, false, false);
    for trial = 1 : numTrial 
        % Get Time variables
        TRON_time = data_behav{session}{trial,1}(1);
        Attack_time_r = data_behav{session}{trial,4}(1);
        nearAttackIRindex = find(data_behav{session}{trial,2}(:,1) < Attack_time_r, 1, 'last');
        valid_IROF_time = data_behav{session}{trial,2}(nearAttackIRindex,2) + TRON_time;
        first_LICK_time = data_behav{session}{trial,3}(1) + TRON_time;
    
        % Get Regression Result during the time
        regResult_HE = data{session}(...
            (midPointTimesData{session} >= timewindow(1) + first_LICK_time) &...
            (midPointTimesData{session} < timewindow(2) + first_LICK_time),:);

        regResult_HW = data{session}(...
            (midPointTimesData{session} >= timewindow(1) + valid_IROF_time) &...
            (midPointTimesData{session} < timewindow(2) + valid_IROF_time),:);

        % some sessions' first and last few trial data are cropped.
        if size(regResult_HE,1) ~= numDatapoints | size(regResult_HW,1) ~= numDatapoints
            fprintf('Session : %02d | trial : %02d data has wronge size of %03d, %03d\n',...
                session, trial, size(regResult_HE,1), size(regResult_HW,1));
            continue;
        end
        decodingError.HE.(behaviorResult(trial)) = [decodingError.HE.(behaviorResult(trial));...
            (regResult_HE(:,3) - regResult_HE(:,5))'*px2cm];

        decodingError.HW.(behaviorResult(trial)) = [decodingError.HW.(behaviorResult(trial));...
            (regResult_HW(:,3) - regResult_HW(:,5))'*px2cm];
    end
end

figure('Name', 'Decoding Error at HE')
subplot(1,1,1);
hold on;
plot(mean(decodingError.HE.A,1), 'b', 'LineWidth',2);
plot(mean(decodingError.HE.E,1), 'r', 'LineWidth',2);
line([numDatapoints/2, numDatapoints/2], [-100, 100], 'Color', 'k', 'LineStyle', '--');
line([0, numDatapoints], [0, 0], 'Color', 'k', 'LineStyle', '--');
ylim([-20, 20]);
xticks(linspace(0, numDatapoints, 5));
xticklabels(string(linspace(timewindow(1), timewindow(2), 5)));
xlabel('Time(s)');
ylabel('Error_{actual distance - predicted distance}(cm)')
legend('Avoid Trials', 'Escape Trials');
title('Decoding Error at HE');

figure('Name', 'Decoding Error at HW')
subplot(1,1,1);
hold on;
plot(mean(decodingError.HW.A,1), 'b', 'LineWidth',2);
plot(mean(decodingError.HW.E,1), 'r', 'LineWidth',2);
line([numDatapoints/2, numDatapoints/2], [-100, 100], 'Color', 'k', 'LineStyle', '--');
line([0, numDatapoints], [0, 0], 'Color', 'k', 'LineStyle', '--');
ylim([-20, 20]);
xticks(linspace(0, numDatapoints, 5));
xticklabels(string(linspace(timewindow(1), timewindow(2), 5)));
xlabel('Time(s)');
ylabel('Error_{actual distance - predicted distance}(cm)')
legend('Avoid Trials', 'Escape Trials');
title('Decoding Error at HW');

%% Decoding error during whole trial
% Show Decoding error across the whole trial

timewindow = [-3, 3];

decodingError.beforeTRON.A = zeros(0, -timewindow(1) * neural_data_rate); % 
decodingError.beforeTRON.E = zeros(0, -timewindow(1) * neural_data_rate); % 
decodingError.TRON2HE.A = zeros(0, 240); % 12 sec
decodingError.TRON2HE.E = zeros(0, 240); % 12 sec
decodingError.HE2HW.A = zeros(0, 120); % 6 sec
decodingError.HE2HW.E = zeros(0, 120); % 6 sec
decodingError.afterHW.A = zeros(0, timewindow(2) * neural_data_rate); % 3 sec
decodingError.afterHW.E = zeros(0, timewindow(2) * neural_data_rate); % 3 sec

for session = 1 : 40
    numTrial = size(data_behav{session}, 1);
    behaviorResult = analyticValueExtractor(data_behav{session}, false, false);
    for trial = 1 : numTrial 
        % Get Time variables
        TRON_time = data_behav{session}{trial,1}(1);
        Attack_time_r = data_behav{session}{trial,4}(1);
        nearAttackIRindex = find(data_behav{session}{trial,2}(:,1) < Attack_time_r, 1, 'last');
        valid_IROF_time = data_behav{session}{trial,2}(nearAttackIRindex,2) + TRON_time;
        first_LICK_time = data_behav{session}{trial,3}(1) + TRON_time;
    
        % Get Regression Result during the time
        regResult_BeforeTRON = data{session}(...
            (midPointTimesData{session} >= timewindow(1) + TRON_time) &...
            (midPointTimesData{session} < TRON_time),:);

        regResult_TRON2HE = data{session}(...
            (midPointTimesData{session} >= TRON_time) &...
            (midPointTimesData{session} < first_LICK_time),:);

        regResult_HE2HW = data{session}(...
            (midPointTimesData{session} >= first_LICK_time) &...
            (midPointTimesData{session} < valid_IROF_time),:);

        regResult_AfterHW = data{session}(...
            (midPointTimesData{session} >= valid_IROF_time) &...
            (midPointTimesData{session} < timewindow(2) + valid_IROF_time),:);

        % some sessions' first and last few trial data are cropped.
        if size(regResult_BeforeTRON,1) ~= -timewindow(1)*neural_data_rate | size(regResult_AfterHW,1) ~= timewindow(2)*neural_data_rate
            fprintf('Session : %02d | trial : %02d data has wronge size.\n',...
                session, trial);
            continue;
        end

        decodingError.beforeTRON.(behaviorResult(trial)) = [...
            decodingError.beforeTRON.(behaviorResult(trial));...
            (regResult_BeforeTRON(:,3) - regResult_BeforeTRON(:,5))' * px2cm];

        decodingError.TRON2HE.(behaviorResult(trial)) = [...
            decodingError.TRON2HE.(behaviorResult(trial));...
            interp1(...
                1:size(regResult_TRON2HE,1),...
                (regResult_TRON2HE(:,3) - regResult_TRON2HE(:,5))' * px2cm,...
                linspace(1,size(regResult_TRON2HE,1),240)...
               )...
            ];

        decodingError.HE2HW.(behaviorResult(trial)) = [...
            decodingError.HE2HW.(behaviorResult(trial));...
            interp1(...
                1:size(regResult_HE2HW,1),...
                (regResult_HE2HW(:,3) - regResult_HE2HW(:,5))' * px2cm,...
                linspace(1,size(regResult_HE2HW,1),120)...
               )...
            ];

        decodingError.afterHW.(behaviorResult(trial)) = [...
            decodingError.afterHW.(behaviorResult(trial));...
            (regResult_AfterHW(:,3) - regResult_AfterHW(:,5))' * px2cm];
    end
end

figure('Name', 'Decoding Error')
subplot(1,1,1);
hold on;
plot([...
    mean(decodingError.beforeTRON.A, 1), ...
    mean(decodingError.TRON2HE.A,1),...
    mean(decodingError.HE2HW.A,1),...
    mean(decodingError.afterHW.A,1)...
    ], 'b', 'LineWidth',2);
plot([...
    mean(decodingError.beforeTRON.E, 1), ...
    mean(decodingError.TRON2HE.E,1),...
    mean(decodingError.HE2HW.E,1),...
    mean(decodingError.afterHW.E,1)...
    ], 'r', 'LineWidth',2);

line([-timewindow(1)*neural_data_rate, -timewindow(1)*neural_data_rate], [-100, 100], 'Color', 'k', 'LineStyle', '--');
line([-timewindow(1)*neural_data_rate + 240, -timewindow(1)*neural_data_rate + 240], [-100, 100], 'Color', 'k', 'LineStyle', '--');
line([-timewindow(1)*neural_data_rate + 360, -timewindow(1)*neural_data_rate + 360], [-100, 100], 'Color', 'k', 'LineStyle', '--');

line([0, 480], [0, 0], 'Color', 'k', 'LineStyle', '--');
ylim([-20, 20]);

xticks(0 : neural_data_rate:480);
xticklabels(string(0 : 24));

text(-timewindow(1)*neural_data_rate, 20, 'TRON', 'HorizontalAlignment','center');
text(-timewindow(1)*neural_data_rate + 240, 20, 'HE', 'HorizontalAlignment','center');
text(-timewindow(1)*neural_data_rate + 360, 20, 'HW', 'HorizontalAlignment','center');

xlabel('Time(s)');
ylabel('Error_{actual distance - predicted distance}(cm)')
legend('Avoid Trials', 'Escape Trials');
title('Decoding Error');

%% Divide Decoding Error in Nest zone, during engaged vs not engaged trial
L1Error_Engaged_Non_Engaged = zeros(40,2);
for session = 1 : 40
    Nest_engaged = [];
    Nest_not_engaged = [];
    numTrial = size(data_behav{session}, 1);
    for trial = 2 : numTrial 
        % Get time variables
        TRON_time = data_behav{session}{trial,1}(1); % in sec, absolute
        last_TROF_time = data_behav{session}{trial-1,1}(2); % in sec, absolute
        latencyToHeadEntry = data_behav{session}{trial,2}(1); % first IR ON Time, in sec, relative
        TROF2IRONInterval = ...
            (data_behav{session}{trial,1}(1) - data_behav{session}{trial-1,1}(2))...
            + data_behav{session}{trial,2}(1); % from Door Close to the first IR, relative under 12 include 50.2% of the trials

        % Get Target WholeTestResult
        %   data during last TROF to current first IRON
        targetResult = data{session}(last_TROF_time <= midPointTimesData{session} & midPointTimesData{session} < TRON_time + latencyToHeadEntry,:);

        %   only select data where animal is in the next zone (col < 225)
        if TROF2IRONInterval >= 12 
            % not engaged trial
            Nest_not_engaged = [Nest_not_engaged; targetResult(targetResult(:,2) < 225, :)];
        else
            % engaged trial
            Nest_engaged = [Nest_engaged; targetResult(targetResult(:,2) < 225, :)];
        end
    end
    if isempty(Nest_engaged) | isempty(Nest_not_engaged)
        continue;
    end
    fprintf("Session : %d - Enaged %d Wander %d\n", session, size(Nest_engaged,1), size(Nest_not_engaged,1));
    L1Error_Engaged_Non_Engaged(session, :) = [...
        mean(abs(Nest_engaged(:,3) - Nest_engaged(:,4))) * px2cm,...
        mean(abs(Nest_not_engaged(:,3) - Nest_not_engaged(:,4))) * px2cm];
end

%% Check Accuracy of the Engaged removed FD Result
basePath = 'D:\Data\Lobster\FineDistanceResult_NestingVariation\FineDistanceResult_rmWander';
px2cm = 0.169;
truncatedTimes_s = 10;
neural_data_rate = 20;

filelist = dir(basePath);
sessionPaths = regexp({filelist.name},'^#\S*.mat','match');
sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));
data = cell(40,1);
result_rmEngaged = table(zeros(40,1), zeros(40,1), 'VariableNames',["Shuffled", "Predicted"]);

for session = 1 : 40
    TANK_name = cell2mat(sessionPaths{session});
    TANK_location = char(strcat(basePath, filesep, TANK_name));
    load(TANK_location, "WholeTestResult"); % PFITestResult, WholeTestResult(row, col, true d , shuffled d, pred d)
    data{session} = WholeTestResult;
    result_rmEngaged.Shuffled(session) = mean(abs(WholeTestResult(:,3) - WholeTestResult(:,4))) * px2cm;
    result_rmEngaged.Predicted(session) = mean(abs(WholeTestResult(:,3) - WholeTestResult(:,5))) * px2cm;
    midPointTimesData{session} = truncatedTimes_s + (1/neural_data_rate)*(0:size(WholeTestResult,1)-1) + 0.5 * (1/neural_data_rate);
end

%% Get Moving Distance
px2cm = 0.169;
output = zeros(numel(sessionPaths),1);
for session = 1 : numel(sessionPaths)
    % Load Position Data
    positionData = data{session}(:, 1:2);
    
    movingDistanceM = sum((diff(positionData(:,1)).^2 + diff(positionData(:,2)).^2).^0.5 .* px2cm)/100;
    output(session) = movingDistanceM;
    
    fprintf("[%d] / %d Complete\n", session, size(sessionPaths,2));
end

%% Draw loss curve
clf;
plot(squeeze(train_log(1,1:(find(squeeze(train_log(1,:,1)) ==0,1)-1),2)), 'k', 'LineWidth',1);
hold on;
plot(squeeze(train_log(1,1:(find(squeeze(train_log(1,:,1)) ==0,1)-1),4)), 'k', 'LineWidth',1, 'LineStyle', ':');
legend({'train', 'test'});
xlabel('epoch');
ylabel('loss (MSE)');
xlim([1,10000]);
set(gca, 'YScale', 'log');
ylim([1000, 100000]);
set(gca,'FontName','Noto Sans');
