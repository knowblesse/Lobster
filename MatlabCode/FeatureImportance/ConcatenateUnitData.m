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


%% Feature Importance - Event Classifier
FI_Event = zeros(size(Unit,1),1);
Unit = [Unit, table(FI_Event, 'VariableNames', {'FI_Event'})];
resultPath = 'D:\Data\Lobster\EventClassificationResult_4C\Output_AE_RFE_max_FI.mat';

load(resultPath);

sessionNames = string(sessionNames);

for session = 1 : 40
    unit = result{session}.importanceUnit_HW;
    for i_unit = 1 : numel(unit)
        %Unit.FI_Event(Unit.Session == sessionNames(session) & Unit.Cell == unit(i_unit)+1) = ...
        %    result{session}.importanceScore_HW(i_unit);% / (result{session}.balanced_accuracy_HW(3) - result{session}.balanced_accuracy_HW(1));
        Unit.FI_Event(Unit.Session == sessionNames(session) & Unit.Cell == unit(i_unit)+1) = 1;
    end
end




