%% ClassifyUnits
% Classify units into groups

%% Load All Unit Data
dataset = load('..\AllUnitData.mat');
Unit = dataset.output;
clearvars -except Unit;

%% Extract zscores
zscoreMatrixHE = zeros(632, 80);
zscoreMatrixHW = zeros(632, 80);
zscoreMatrixHW_A = zeros(632, 80);
zscoreMatrixHW_E = zeros(632, 80);

for cell = 1 : 632
    zscoreMatrixHE(cell,:) = Unit.Zscore{cell}.first_LICK;
    zscoreMatrixHW(cell,:) = Unit.Zscore{cell}.valid_IROF;
    zscoreMatrixHW_A(cell,:) = Unit.Zscore{cell}.valid_IROF_A;
    zscoreMatrixHW_E(cell,:) = Unit.Zscore{cell}.valid_IROF_E;
end

%% Sort units       

[groupingResult_HE, ~] = groupUnits(zscoreMatrixHE, 'numCluster', 8, 'cutoffLimit', 50, 'showGraph', true);
[groupingResult_HW, numGroup] = groupUnits(zscoreMatrixHW, 'numCluster', 8, 'cutoffLimit', 50, 'showGraph', true);
[groupingResult_HW_A, numGroup] = groupUnits(zscoreMatrixHW_A, 'numCluster', 8, 'cutoffLimit', 50, 'showGraph', true);
[groupingResult_HW_E, numGroup] = groupUnits(zscoreMatrixHW_E, 'numCluster', 8, 'cutoffLimit', 50, 'showGraph', true);


%% Add to the Unit Data

groupData = table(...
    groupingResult_HE,...
    groupingResult_HW,...
    groupingResult_HW_A,...
    groupingResult_HW_E,...
    'VariableNames',["Group_HE", "Group_HW", "Group_HW_A", "Group_HW_E"]);

Unit = [Unit, groupData];
