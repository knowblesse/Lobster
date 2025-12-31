%% review_program
% Scripts for testing feature importance

px2cm = 0.169;

%% Feature Importance - Fine Distance Regressor
resultPath = 'D:\Data\Lobster\FineDistanceResult_syncFixed_May';
filelist = dir(resultPath);
sessionPaths = regexp({filelist.name},'^#\S*.mat','match');
sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));


%%
fprintf('session : 00/40');
sessionPFI = cell(40,1);
for session = 1 : 40
    sessionName = cell2mat(regexp(cell2mat(sessionPaths{session}), '^#.*?L', 'match'));
    MAT_filename = cell2mat(sessionPaths{session});
    MAT_filePath = char(strcat(resultPath, filesep, MAT_filename));
    load(MAT_filePath); % PFITestResult, WholeTestResult(row, col, true d , shuffled d, pred d)
    
    % calculate FI for every units
    err_noace = mean(abs(WholeTestResult(:,3) - WholeTestResult(:,5))) * px2cm;
    err_control = mean(abs(WholeTestResult(:,3) - WholeTestResult(:,4))) * px2cm;
    
    numUnit = size(PFITestResult, 2);
    numIteration = size(PFITestResult, 3);
    
    sessionPFI_single = zeros(numIteration, numUnit);
    
    for unit = 1 : numUnit
        err_corrupted = mean(squeeze(abs(WholeTestResult(:,3) - PFITestResult(:,unit, :))), 1) * px2cm;
        sessionPFI_single(:, unit) = err_corrupted';
    end
    sessionPFI{session} = sessionPFI_single;
    fprintf('\b\b\b\b\b%02d/40', session);
end

%% Read Unit table, get FI rank for distance regressor, save FI order
base_path = "D:\Data\Lobster\FineDistanceDataset";
sessionNames = unique(Unit.Session);
for session = 1 : 40
    [~, seq] = sort(Unit.FI_Distance(Unit.Session == sessionNames(session)));
    writematrix(seq, base_path + filesep + sessionNames(session) + filesep + 'FI_rank.csv');
end

%% Read Unit table,get FI rank for event classifier, save FI order
base_path = "D:\Data\Lobster\EventClassificationData_4C";
sessionNames = unique(Unit.Session);
for session = 1 : 40
    % Small value in cross entropy (ex. -10) means low error. 
    [~, seq] = sort(Unit.FI_EC_ACC(Unit.Session == sessionNames(session)));
    % Save with X and y data
    save(...
        strcat(base_path, filesep, sessionNames(session), '_eventClassificationData.mat'),...
        'seq',...
        '-append');
end
    
%% Feature Importance - using ace (top 20% neurons with high FI)
resultPath = 'D:\Data\Lobster\FineDistanceResult_syncFixed_240501';

filelist = dir(resultPath);
sessionPaths = regexp({filelist.name},'^#\S*.mat','match');
sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));

fprintf('session : 00/40');
output_result = zeros(40, 3);
for session = 1 : 40
    sessionName = cell2mat(regexp(cell2mat(sessionPaths{session}), '^#.*?L', 'match'));
    MAT_filename = cell2mat(sessionPaths{session});
    MAT_filePath = char(strcat(resultPath, filesep, MAT_filename));
    load(MAT_filePath); % PFITestResult, WholeTestResult(row, col, true d , shuffled d, pred d)
    
    % calculate FI for every units
    err_shuffle = mean(abs(WholeTestResult(:,3) - WholeTestResult(:,4))) * px2cm;
    err_noace = mean(abs(WholeTestResult(:,3) - WholeTestResult(:,5))) * px2cm;
    err_control = mean(abs(WholeTestResult(:,3) - WholeTestResult(:,6))) * px2cm;
    
    output_result(session, :) = [err_shuffle, err_noace, err_control];
       
    fprintf('\b\b\b\b\b%02d/40', session);
end

%% Feature Importance - Event - using ace (top 20% neurons with high FI)
resultPath = 'D:\Data\Lobster\BNB_Result_unitshffle_noace.mat';

load(resultPath);

output_result = zeros(40, 2);
for session = 1 : 40
    output_result(session, :) = result{session}.balanced_accuracy_HWAE;
end
