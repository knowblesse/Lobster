%% AnalyzeDistanceRegressionData

basePath = 'D:\Data\Lobster\DistanceRegressionResult';

filelist = dir(basePath);
sessionPaths = regexp({filelist.name},'^#\S*.mat','match');
sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));

load("Apparatus.mat");

%% Load Data by session
data = cell(1,40);
FI = cell(1,40);
for session = 1 : 40
    TANK_name = cell2mat(sessionPaths{session});
    TANK_location = char(strcat(basePath, filesep, TANK_name));
    load(TANK_location); % PFITestResult, WholeTestResult(row, col, true d , shuffled d, pred d)
    
    data{session} = WholeTestResult;
    FI{session} = PFITestResult;
end

%% Compare Error btw shuffled and predicted
result1 = table(zeros(40,1), zeros(40,1), 'VariableNames',["Shuffled", "Predicted"]);
for session = 1 : 40
    result1.Shuffled(session) = mean(abs(data{session}(:,3) - data{session}(:,4)));
    result1.Predicted(session) = mean(abs(data{session}(:,3) - data{session}(:,5)));
end

%% Compare Error btw Nesting zone and Foraging zone
result2 = table(zeros(40,1), zeros(40,1), 'VariableNames', ["NestError", "ForagingError"]);
for session = 1 : 40
    nestError = [];
    foragingError = [];
    for dataIndex = 1 : size(data{session}, 1)
        datapoint = data{session}(dataIndex, :);
        if apparatus.mask(round(datapoint(1)), round(datapoint(2))) == 1
            if datapoint(2) < 200
                nestError = [nestError, locError{session}(dataIndex)];
            else
                foragingError = [foragingError, locError{session}(dataIndex)];
            end
        end
    end
    result2.NestError(session) = mean(nestError);
    result2.ForagingError(session) = mean(foragingError);
end



