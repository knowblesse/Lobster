%% Script2FindNewCluster

%% New Group and label
output = [output, table(responsive(:, 1), 'VariableNames', "responsive_HE")];
output = [output, table(responsive(:, 4), 'VariableNames', "responsive_HW")];

[h, p] = ttest2(... 
    Unit.FI_Distance(any(abs(first_LICK_zscores) > 4,2)),...
    Unit.FI_Distance(~any(abs(first_LICK_zscores) > 4,2)))

[h, p] = ttest2(...% significant
    Unit.FI_Distance(any(abs(valid_IROF_zscores) > 4,2)),...
    Unit.FI_Distance(~any(abs(valid_IROF_zscores) > 4,2)))

[h, p] = ttest2(...
    Unit.FI_Distance(any((valid_IROF_zscores) > 4,2)),...
    Unit.FI_Distance(~any((valid_IROF_zscores) > 4,2)))

[h, p] = ttest2(...
    Unit.FI_Distance((mean(valid_IROF_zscores(:, 1:30),2) >  mean(valid_IROF_zscores(:, 51:80),2))),...
    Unit.FI_Distance(~(mean(valid_IROF_zscores(:, 1:30),2) >  mean(valid_IROF_zscores(:, 51:80),2))))

[h, p] = ttest2(...
    Unit.FI_Event_Score_Relative(Unit.FI_Event_Score > 0 & any(abs(valid_IROF_zscores) > 4,2)),...
    Unit.FI_Event_Score_Relative(Unit.FI_Event_Score > 0 & ~any(abs(valid_IROF_zscores) > 4,2)))

%% Clustering Method
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

%% HE
zscoreMatrix = zscoreMatrixHE;
numCellCluster = 8;

Z_ = linkage(zscoreMatrix, 'average', 'correlation');
Y_ = pdist(zscoreMatrix, 'correlation');
ici_ = inconsistent(Z_, 130);
figure(1);
hist(ici_(:,4));
title('Inconsistency Coef');

CellType_ = cluster(Z_, 'maxclust', numCellCluster);

figure(2);
%clf;

subplot(1,2,2);
hold on;

lines = [];
labels = {};
for clstr = 1 : numCellCluster
%     if ~ismember(clstr, [2,4,6])
%         continue;
%     end
    linethickness = sum(CellType_ == clstr)/100;
    [~, l_, ~] = shadeplot(zscoreMatrix(CellType_ == clstr, :), 'SD', 'sem', 'LineWidth', linethickness);
    lines = [lines, l_];
    labels = [labels, sprintf('Type %d n=%d', clstr, sum(CellType_ == clstr))];
end
legend(lines, labels);
%%
figure(3);
clf;
dendrogram(Z_, 30, "Reorder", optimalleaforder(Z_, Y_, 'Criteria','group'), "ColorThreshold",mean(Z_(end-numCellCluster+1:end-numCellCluster+2,3)));

figure(4);
clf;
dendrogram(Z_, 632, "Reorder", optimalleaforder(Z_, Y_, 'Criteria','group'), "ColorThreshold",mean(Z_(end-numCellCluster+1:end-numCellCluster+2,3)));

figure(5);
clf;
lines = dendrogram(Z_, 632, "Reorder", optimalleaforder(Z_, Y_, 'Criteria','group'));
for i = 1 : 631
    lines(i).Color = 'k';
     pause(0.01);
     drawnow;
end

%% Label Table
zscoreMatrix = zscoreMatrixHE;
numCellCluster = 8;

Z_ = linkage(zscoreMatrix, 'average', 'correlation');

CellType_ = cluster(Z_, 'maxclust', numCellCluster);

HE_Type = zeros(632, 1);
HE_Type(CellType_ == 2) = 1; % reward
HE_Type(CellType_ == 4) = 2; % anticipatory

Unit = [Unit, table(HE_Type, 'VariableNames',{'HEClass'})];

% -------------- HW---------------
zscoreMatrix = zscoreMatrixHW;
numCellCluster = 8;

Z_ = linkage(zscoreMatrix, 'average', 'correlation');

CellType_ = cluster(Z_, 'maxclust', numCellCluster);

HW_Type = zeros(632, 1);
HW_Type(CellType_ == 5) = 1; % Decreased After HW
HW_Type(CellType_ == 7) = 2; % Inhibition on HW
HW_Type(CellType_ == 8) = 3; % Slight Increase After HW

Unit = [Unit, table(HW_Type, 'VariableNames',{'HWClass'})];

% -------------- HW_A---------------
zscoreMatrix = zscoreMatrixHW_A;
numCellCluster = 8;

Z_ = linkage(zscoreMatrix, 'average', 'correlation');

CellType_ = cluster(Z_, 'maxclust', numCellCluster);

HW_A_Type = zeros(632, 1);
HW_A_Type(CellType_ == 1) = 1; % Slight Increase After HW_A
HW_A_Type(CellType_ == 2) = 2; % Peak on HW_A
HW_A_Type(CellType_ == 7) = 3; % Back to normal after HW_A

Unit = [Unit, table(HW_A_Type, 'VariableNames',{'HW_A_Class'})];

% -------------- HW_E---------------
zscoreMatrix = zscoreMatrixHW_E;
numCellCluster = 8;

Z_ = linkage(zscoreMatrix, 'average', 'correlation');

CellType_ = cluster(Z_, 'maxclust', numCellCluster);

HW_E_Type = zeros(632, 1);
HW_E_Type(CellType_ == 2) = 1; % Peak on HW_E
HW_E_Type(CellType_ == 4) = 2; % Back to normal after HW_E
HW_E_Type(CellType_ == 6) = 3; % Inhibition on HW_E

Unit = [Unit, table(HW_E_Type, 'VariableNames',{'HW_E_Class'})];





%%

[h, p] = ttest2(... 
    Unit.FI_Distance(Unit.HW_E_Class == 2),...
    Unit.FI_Distance(Unit.HW_E_Class == 3))

% Event Classifier에 Imporance cell의 비중

% Event_Score_Relative
% HE : 2 anticipatory
% HW : 
% HW_A : 
% HW_E : 

% Distance
% HE : 2 anticipatory
% HW : 1-3
% HW_A : 0-1 0-2 0-3
% HW_E : 0-1 (0-3)





% 
% 
% % based on pre mid post
% prePoints = mean(valid_IROF_E_zscores(:,1:38), 2);
% postPoints = mean(valid_IROF_E_zscores(:,43:80), 2);
% 
% figure(1);
% clf;
% subplot(1,1,1);
% hold on;
% plot(mean(valid_IROF_zscores(...
%     prePoints - 0.5 > postPoints...
%     , :), 1));
% plot(mean(valid_IROF_zscores(...
%     prePoints + 0.5 < postPoints...
%     , :), 1))
% 
% 
% Unit = removevars(Unit, 'inhibition');
% Unit = removevars(Unit, 'excitation');
% Unit = [Unit, table(prePoints - 0.5 > postPoints, 'VariableNames', "inhibition")];
% Unit = [Unit, table(prePoints + 0.5 < postPoints, 'VariableNames', "excitation")];
% 
% Unit.FI_Distance(Unit.excitation == 1)
% Unit.FI_Distance(Unit.inhibition == 1)
% Unit.FI_Distance(Unit.inhibition == 0 & Unit.excitation == 0)
