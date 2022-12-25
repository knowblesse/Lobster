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
Unit = [Unit, table(zeros(size(Unit,1),1), zeros(size(Unit,1),1), zeros(size(Unit,1),1), 'VariableNames', {'FI_Event_Importance', 'FI_Event_Score', 'FI_Event_Score_Relative'})];
resultPath = 'D:\Data\Lobster\EventClassificationResult_4C\Output_AE_RFE_max_FI.mat';

load(resultPath);

sessionNames = string(sessionNames);

for session = 1 : 40
    unit = result{session}.importanceUnit_HW;
    for i_unit = 1 : numel(unit)
        Unit.FI_Event_Importance(Unit.Session == sessionNames(session) & Unit.Cell == unit(i_unit)+1) = 1;

        Unit.FI_Event_Score(Unit.Session == sessionNames(session) & Unit.Cell == unit(i_unit)+1) = ...
            result{session}.importanceScore_HW(i_unit);

        Unit.FI_Event_Score_Relative(Unit.Session == sessionNames(session) & Unit.Cell == unit(i_unit)+1) = ...
            result{session}.importanceScore_HW(i_unit) / (result{session}.balanced_accuracy_HW(3) - result{session}.balanced_accuracy_HW(1));
    end
end

%% 

[h, p] = ttest2(...
    Unit.FI_Event_Score((Unit.HEClass == 1) & (Unit.EC_Score(:,2) > 0.7) & (Unit.FI_Event_Importance > 0)),...
    Unit.FI_Event_Score((Unit.HEClass == 2) & (Unit.EC_Score(:,2) > 0.7) & (Unit.FI_Event_Importance > 0)))



