%% ClassifyUnits
% Classify units into groups

%% Load All Unit Data
dataset = load('..\AllUnitData.mat');
Unit = dataset.output;

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

%% Generate Table
drawConfusionMatrix(groupingResult_HE, groupingResult_HW, 'ShowSum', true, 'ColName', "HW Group", "RowName", "HE Group", "Color", [0, 0, 0]);

%% Draw HE1-HW1 and HE2-HW2 graph

groupingResult = zeros(size(Unit,1),1);
groupingResult(groupData.Group_HE == 1 & groupData.Group_HW == 1) = 1;
groupingResult(groupData.Group_HE == 2 & groupData.Group_HW == 2) = 2;

colors_bw = [0, 0, 0; 0.5, 0.5, 0.5];
for matrix = {zscoreMatrixHE, zscoreMatrixHW}
    zscoreMatrix = matrix{1};

    fig = figure();
    axes();
    hold on;
    lines = [];
    legends = {};
    for group = 1 : 2
        [~, obj_line, ~] = shadeplot(...
            zscoreMatrix(groupingResult == group, :),...
            'SD', 'sem',... %'LineWidth', sum(groupingResult == group)/100,...
            'LineWidth', 1.3,...
            'FaceAlpha', 0.3,...
            'Color', colors_bw(group,:));
        lines = [lines, obj_line];
    end
    line(xlim, [0,0], 'LineStyle', ':', 'Color', [0.3, 0.3, 0.3]);
    ylabel('Z score');
    xlabel('Time (sec)');
    xticks(0:20:80);
    xticklabels(-2:2);
    legend(lines, {'HE1-HW1', 'HE2-HW2'}, 'FontSize', 6.6);
    set(gca, 'FontName', 'Noto Sans');
    pos = get(gcf, 'Position');
    set(gcf, 'Position', [pos(1), pos(2), 288, 236]);
end


