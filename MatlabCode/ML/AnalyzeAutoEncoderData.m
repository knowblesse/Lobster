%% AnalyzeAutoEncoderData
basePath_autoencoder = 'D:\Data\Lobster\AutoEncoderResult';
basePath_other = 'D:\Data\Lobster\FineDistanceResult_speed';

filelist = dir(basePath_autoencoder);
sessionPaths = regexp({filelist.name},'^#\S*.mat','match');
sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));

load("Apparatus.mat");
px2cm = 0.169;
truncatedTimes_s = 10;
neural_data_rate = 20;

%% Load Data by session
numSession = numel(sessionPaths);
data_autoencoder = cell(1,numSession);
midPointTimesData = cell(1,numSession);
numCell = zeros(numSession, 1);
for session = 1 : numSession
    TANK_name = cell2mat(sessionPaths{session});
    TANK_location = char(strcat(basePath_autoencoder, filesep, TANK_name));
    load(TANK_location, "WholeTestResult", "midPointTimes"); % PFITestResult, WholeTestResult
    data_autoencoder{session} = [WholeTestResult, sum((WholeTestResult(:,1:2)-[280, 640]) .^2,2).^0.5];
    %data_autoencoder{session} = WholeTestResult;
    if exist("midPointTimes") == 0
            midPointTimesData{session} = truncatedTimes_s + (1/neural_data_rate)*(0:size(WholeTestResult,1)-1) + 0.5 * (1/neural_data_rate);
    else
        midPointTimesData{session} = midPointTimes;
    end
    
    %% other
%     regresult = regexp(TANK_name, '.*?L', 'match');
%     regresult = regresult{1};
%     load(glob(basePath_other, strcat(regresult, '.*'), true));
%     data_autoencoder{session} = [data_autoencoder{session}, WholeTestResult(:,3)];
end

%% Check correlation
result = zeros(40,1);
for session = 1 : numSession
    corrResult = corrcoef(data_autoencoder{session}(:,3:end));
    result(session) = max(abs(corrResult(end,1:end-1)));
end
