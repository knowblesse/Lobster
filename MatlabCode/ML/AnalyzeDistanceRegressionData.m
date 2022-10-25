%% AnalyzeDistanceRegressionData

basePath = 'D:\Data\Lobster\DistanceRegressionResult';

filelist = dir(basePath);
sessionPaths = regexp({filelist.name},'^#\S*.csv','match');
sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));

load("Apparatus.mat");

%% Load Data by session
data = cell(1,40);
locError = cell(1,40);
for session = 1 : 40
    TANK_name = cell2mat(sessionPaths{session});
    TANK_location = char(strcat(basePath, filesep, TANK_name));

    otherTank = regexp(TANK_name, '(?<f1>.*?)_distance_.*', 'names');

    xyPosition = readmatrix(fullfile('D:\Data\Lobster\LocationRegressionResult', strcat(otherTank.f1, 'result.csv')));

    data{session} = [xyPosition(:,1:2), readmatrix(TANK_location)];
    locError{session} = abs(data{session}(:,3) - data{session}(:,5));
end


%% Compare Error btw Nesting zone and Foraging zone
result = table(zeros(40,1), zeros(40,1), 'VariableNames', ["NestError", "ForagingError"]);
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
    result.NestError(session) = mean(nestError);
    result.ForagingError(session) = mean(foragingError);
end



