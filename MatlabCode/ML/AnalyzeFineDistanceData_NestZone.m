%% AnalyzeFineDistanceData_NestZone
% Find why nest zone has high error

basePath = 'D:\Data\Lobster\FineDistanceResult_syncFixed';
behavDataPath = 'D:\Data\Lobster\BehaviorData';
datasetDataPath = 'D:\Data\Lobster\FineDistanceDataset';

filelist = dir(basePath);
sessionPaths = regexp({filelist.name},'^#\S*.mat','match');
sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));

load("Apparatus.mat");
px2cm = 0.169;
truncatedTimes_s = 10;
neural_data_rate = 20;

%% Load Behavior Data
data_behav = cell(1,40);
for session = 1 : 40
    TANK_name = cell2mat(sessionPaths{session});
    TANK_location = char(strcat(basePath, filesep, TANK_name));
    load(fullfile(behavDataPath, strcat(cell2mat(regexp(TANK_name, '#.*?[PI]L', 'match')), '.mat')));
    data_behav{session} = ParsedData;
end

%% Load Data by session
data = cell(1,40);
midPointTimes = cell(1,40);
for session = 1 : 40
    TANK_name = cell2mat(sessionPaths{session});
    TANK_location = char(strcat(basePath, filesep, TANK_name));
    load(TANK_location); % PFITestResult, WholeTestResult(row, col, true d , shuffled d, pred d)
    data{session} = WholeTestResult;
    warning('Currently midPointTimes are not loaded, but caculated');
    midPointTimes{session} = truncatedTimes_s + (1/neural_data_rate)*(0:size(WholeTestResult,1)-1) + 0.5 * (1/neural_data_rate);
end

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

        % Get Target WholeTestResult
        %   data during last TROF to current first IRON
        targetResult = data{session}(last_TROF_time <= midPointTimes{session} & midPointTimes{session} < TRON_time + latencyToHeadEntry,:);

        %   only select data where animal is in the next zone (col < 225)
        if latencyToHeadEntry >= 5 
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
    L1Error_Engaged_Non_Engaged(session, :) = [...
        mean(abs(Nest_engaged(:,3) - Nest_engaged(:,4))) * px2cm,...
        mean(abs(Nest_not_engaged(:,3) - Nest_not_engaged(:,4))) * px2cm];
end

%% Check Accuracy of the Engaged removed FD Result
basePath = 'C:\Users\knowb\Desktop\rmWander\FineDistanceResult_rmWander';
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
    load(TANK_location); % PFITestResult, WholeTestResult(row, col, true d , shuffled d, pred d)
    data{session} = WholeTestResult;
    result_rmEngaged.Shuffled(session) = mean(abs(WholeTestResult(:,3) - WholeTestResult(:,4))) * px2cm;
    result_rmEngaged.Predicted(session) = mean(abs(WholeTestResult(:,3) - WholeTestResult(:,5))) * px2cm;
    midPointTimes{session} = truncatedTimes_s + (1/neural_data_rate)*(0:size(WholeTestResult,1)-1) + 0.5 * (1/neural_data_rate);
end


%% Analyze Nest Zone error
nzoneError = [];
figure('Name', 'Nest Zone Errors');
for session = 1 : 40
    subplot(5,8,session);
    WholeTestResult = data{session};
    nzoneError = [nzoneError; WholeTestResult(WholeTestResult(:,2)<225, [3,5])];
    histogram(diff(WholeTestResult(WholeTestResult(:,2)<225, [3,5]), 1, 2));
    title(cell2mat(regexp(sessionPaths{session}{1}, '#.*?[PI]L', 'match')), 'Interpreter', 'none');
end
    
%% Gaussian Mixture Model fitting to the Nestzone Data
figure();
session = 7;
WholeTestResult = data{session};
nestingData = WholeTestResult(WholeTestResult(:,2)<225, :);
errors = diff(nestingData(:, [3,5]), 1, 2);
histogram(errors);

% 
tabulated  = tabulate(round(errors));
GMModel = fitgmdist(round(errors),2,'Options',statset('MaxIter',1000));
criterion = mean(GMModel.ComponentProportion * GMModel.mu);
figure();
bar(tabulated(tabulated(:,1) < criterion,1),tabulated(tabulated(:,1) < criterion,3)/100,'FaceColor','r', 'LineStyle','none');
hold on;
bar(tabulated(tabulated(:,1) >= criterion,1),tabulated(tabulated(:,1) >= criterion,3)/100,'FaceColor','b', 'LineStyle','none');


plot(tabulated(:,1),pdf(GMModel,tabulated(:,1)),'Color','r','LineWidth',1);
y_axis = ylim();

line([criterion, criterion], y_axis, 'Color', 'y', 'LineStyle', ':', 'LineWidth', 2');

% error < -100 애들의 특성은?
figure();
imagesc(apparatus.image);
hold on;
scatter(nestingData(errors < -100, 2), nestingData(errors < -100, 1), '.r');
scatter(nestingData(errors > -100, 2), nestingData(errors > -100, 1), '.b');


