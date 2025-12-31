%% AnalyzeFineDistanceData

basePath = 'D:\Data\Lobster\FineDistanceResult_other_decoders';
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
result1 = table(zeros(numSession,1), zeros(numSession,1), zeros(numSession,1), zeros(numSession,1), 'VariableNames',...
    ["LinearRegression", "QuadraticRegression", "SVMRegressor", "RandomForestRegressor"]);
for session = 1 : numSession
    result1.LinearRegression(session) = mean(abs(data{session}(:,3) - data{session}(:,4))) * px2cm;
    result1.QuadraticRegression(session) = mean(abs(data{session}(:,3) - data{session}(:,5))) * px2cm;
    result1.SVMRegressor(session) = mean(abs(data{session}(:,3) - data{session}(:,6))) * px2cm;
    result1.RandomForestRegressor(session) = mean(abs(data{session}(:,3) - data{session}(:,7))) * px2cm;
end