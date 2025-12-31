%% AnalyzeFineDistanceData

basePath = 'D:\Data\Lobster\FineDistanceResult_syncFixed_first_half';
behavDataPath = 'D:\Data\Lobster\BehaviorData';

filelist = dir(basePath);
sessionPaths = regexp({filelist.name},'^#\S*.mat','match');
sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));

load("Apparatus.mat");
px2cm = 0.169;
truncatedTimes_s = 10;
neural_data_rate = 20;

numSession = numel(sessionPaths);

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