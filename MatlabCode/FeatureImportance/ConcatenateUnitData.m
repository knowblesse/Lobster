%% ConcatenateUnitData
% To create "AllUnitData_PETHincl.mat", 
% run partially run ./Spike/SortedPETH_Scripts.m
load('AllUnitData_PETHincl.mat');
Unit = output;
clearvars output

%% Feature Importance - Fine Distance Regressor
resultPath = 'D:\Data\Lobster\FineDistanceResult';

filelist = dir(resultPath);
sessionPaths = regexp({filelist.name},'^#\S*.mat','match');
sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));

FI_Event = zeros(size(Unit,1),1);

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
        
        FI_factor = (err_corrupted - err_original) / (err_shuffled - err_original);
        FI_Event(Unit.Session == sessionName & Unit.Cell == unit) = FI_factor;
    end
    fprintf('\b\b\b\b\b%02d/40', session);
end

Unit = [Unit, table(FI_Event, 'VariableNames', {'FI_Distance'})];


%% Two Event Classifier Comparison
% Why FI is different?
result1Path = 'D:\Data\Lobster\EventClassificationResult_4C\Output_AE_RFE_max_FI.mat';
result2Path = 'D:\Data\Lobster\CV5_HEC_Result.mat';

result1 = load(result1Path);
result2 = load(result2Path);

score1 = zeros(40,3);
score2 = zeros(40,2);

FI1 = [];
FI2 = [];

for session = 1 : 40
    score1(session, :) = result1.result{session}.balanced_accuracy_HW;
    score2(session, :) = result2.result{session}.balanced_accuracy_HWAE;

    numCell = numel(result1.result{session}.unitRank_HE);
    fi_ = zeros(numCell,1);
    fi_(result1.result{session}.importanceUnit_HW+1) = result1.result{session}.importanceScore_HW;
    FI1 = [FI1; fi_];
    
    FI2 = [FI2; permutation_feature_importance(result2.result{session}.WholeTestResult_HWAE, result2.result{session}.PFITestResult_HWAE)];
end


corr(FI1(Unit.Session ~= "#21JAN5-210803-182450_IL"), FI2(Unit.Session ~= "#21JAN5-210803-182450_IL"))
corr(FI1(FI1 > 0 & Unit.Session ~= "#21JAN5-210803-182450_IL"), FI2(FI1 > 0 & Unit.Session ~= "#21JAN5-210803-182450_IL"))


%% Put Event Classifier Score into Unit table
Unit = [Unit, table(zeros(632,3), 'VariableNames', "EC_Score")];
result1 = load('D:\Data\Lobster\EventClassificationResult_4C\Output_AE_RFE_max_FI.mat');
result1.sessionNames = string(result1.sessionNames);
for session = 1 : 40
    Unit.EC_Score(Unit.Session == result1.sessionNames(session), :) = ...
        repmat(result1.result{session}.balanced_accuracy_HW, sum(Unit.Session == result1.sessionNames(session)), 1);
end



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



