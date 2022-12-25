%% ConcatenateUnitData
% To create "AllUnitData_PETHincl.mat", 
% run partially run ./Spike/SortedPETH_Scripts.m
ClassifyUnits;

clearvars -except Unit

%% Feature Importance - Fine Distance Regressor
resultPath = 'D:\Data\Lobster\FineDistanceResult';

filelist = dir(resultPath);
sessionPaths = regexp({filelist.name},'^#\S*.mat','match');
sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));

FI_Distance_Ratio = zeros(size(Unit,1),1);
FI_Distance_Difference = zeros(size(Unit,1),1);
FI_Distance_Relative = zeros(size(Unit,1),1);

fprintf('session : 00/40');
for session = 1 : 40
    sessionName = cell2mat(regexp(cell2mat(sessionPaths{session}), '^#.*?L', 'match'));
    MAT_filename = cell2mat(sessionPaths{session});
    MAT_filePath = char(strcat(resultPath, filesep, MAT_filename));
    load(MAT_filePath); % PFITestResult, WholeTestResult(row, col, true d , shuffled d, pred d)

    % calculate FI for every units
    err_original = mean(abs(WholeTestResult(:,3) - WholeTestResult(:,5)));
    err_shuffled = mean(abs(WholeTestResult(:,3) - WholeTestResult(:,4)));
    for unit = 1 : size(PFITestResult, 2)
        err_corrupted = mean(mean(abs(WholeTestResult(:,3) - PFITestResult(:,unit, :))));
        
        % FI Factor Ratio
        FI_Distance_Ratio(Unit.Session == sessionName & Unit.Cell == unit) = ...
            err_corrupted / err_original;

        % FI Factor Difference
        FI_Distance_Difference(Unit.Session == sessionName & Unit.Cell == unit) = ...
            err_corrupted - err_original;

        FI_Distance_Relative(Unit.Session == sessionName & Unit.Cell == unit) = ...
            (err_corrupted - err_original) / (err_shuffled - err_original);
    end
    fprintf('\b\b\b\b\b%02d/40', session);
end

Unit = [Unit, table(FI_Distance_Ratio, FI_Distance_Difference, FI_Distance_Relative, 'VariableNames', {'FI_Distance_Ratio', 'FI_Distance_Difference', 'FI_Distance_Relative'})];

%% Feature Importance - Event Classifier
resultPath = 'D:\Data\Lobster\EventClassificationResult_4C\Output_AE_RFE_max_FI.mat';

load(resultPath);

sessionNames = string(sessionNames);

FI_Event_Ratio = [];
FI_Event_Difference = [];
FI_Event_Relative = [];
EC_Score = [];

for session = 1 : 40
    FI_Event_Ratio = [FI_Event_Ratio; permutation_feature_importance(result{session}.WholeTestResult_HWAE, result{session}.PFITestResult_HWAE, ...
        'method', 'ratio')];
    FI_Event_Difference = [FI_Event_Difference; permutation_feature_importance(result{session}.WholeTestResult_HWAE, result{session}.PFITestResult_HWAE, ...
        'method', 'difference')];
    FI_Event_Relative = [FI_Event_Relative; permutation_feature_importance(result{session}.WholeTestResult_HWAE, result{session}.PFITestResult_HWAE, ...
        'method', 'relative')];
    EC_Score = [EC_Score; repmat(result{session}.balanced_accuracy_HWAE(2), size(result{session}.PFITestResult_HWAE,2),1)];
end

Unit = [Unit, table(FI_Event_Ratio, FI_Event_Difference, FI_Event_Relative, EC_Score, 'VariableNames', {'FI_Event_Ratio', 'FI_Event_Difference', 'FI_Event_Relative', 'EC_Score'})];

%% Clip FI
Unit.FI_Distance_Ratio(Unit.FI_Distance_Ratio < 1) = 1; % smaller than 1 means corrupted performs better
Unit.FI_Distance_Difference(Unit.FI_Distance_Ratio < 0) = 0;

Unit.FI_Event_Ratio(Unit.FI_Event_Ratio > 1) = 1; % larger than 1 means corrupted performs better
Unit.FI_Event_Difference(Unit.FI_Event_Difference < 0) = 0;

%% 

[h, p] = ttest2(...
    Unit.FI_Distance_Ratio((Unit.Group_HE == 1) & (Unit.EC_Score > 0.6)),...
    Unit.FI_Distance_Ratio((Unit.Group_HE == 2) & (Unit.EC_Score > 0.6)))



