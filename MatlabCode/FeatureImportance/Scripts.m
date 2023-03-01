%% TestingScripts

%% Load Unit Class Data
ClassifyUnits;
clearvars -except Unit

%% Load Two Data
targetTank = '#21JAN2-210406-190737_IL';
result_distance = load(fullfile('D:\Data\Lobster\FineDistanceResult_syncFixed', ...
    strcat(targetTank, 'result_distance.mat')));
result_event = load('D:\Data\Lobster\BNB_Result_fullshuffle.mat');
result_event = result_event.result{find(string(result_event.sessionNames) == targetTank)};

%% Calculate Unit Importance for distance data
numUnit = size(result_distance.PFITestResult,2);
original_data_error = mean(abs(result_distance.WholeTestResult(:,3) - result_distance.WholeTestResult(:,5))) * 0.169;

one_unit_shuffled_data_errors = zeros(numUnit,1);
for unit = 1 : numUnit
    one_unit_shuffled_data_errors(unit) = mean(abs(result_distance.WholeTestResult(:,3) - result_distance.PFITestResult(:,unit))) * 0.169;
end

UI_distance = one_unit_shuffled_data_errors - original_data_error;

clearvars unit original_data_error one_unit_shuffled_data_errors

%% Calculate Unit Importance for event data
UI_event = permutation_feature_importance(result_event.WholeTestResult_HWAE > 0.5, result_event.PFITestResult_HWAE > 0.5, 'method', 'difference');

%%
UnitData = Unit(Unit.Session == targetTank, :);
UI = [UI_distance, UI_event, UnitData.Group_HE, UnitData.Group_HW, UnitData.Group_HW_A, UnitData.Group_HW_E];
    
