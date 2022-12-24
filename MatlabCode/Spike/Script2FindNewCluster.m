%% Script2FindNewCluster

dataset = load('C:\Users\Knowblesse\SynologyDrive\AllUnitData.mat');
Unit = dataset.output;
clearvars -except Unit;

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

%% Sort units
% -------------- HE---------------
zscoreMatrix = zscoreMatrixHE;
numCellCluster = 8;

Z_ = linkage(zscoreMatrix, 'average', 'correlation');

CellType_ = cluster(Z_, 'maxclust', numCellCluster);

HE_Type = zeros(632, 1);
HE_Type(CellType_ == 2) = 1; % reward
HE_Type(CellType_ == 4) = 2; % anticipatory

Unit = [Unit, table(HE_Type, 'VariableNames',{'HEClass'})];

figure('Name', 'HE');
subplot(1,1,1);
hold on;
shadeplot(zscoreMatrix(HE_Type == 1, :), 'SD', 'sem', 'LineWidth', sum(HE_Type == 1)/100);
shadeplot(zscoreMatrix(HE_Type == 2, :), 'SD', 'sem', 'LineWidth', sum(HE_Type == 2)/100);
ylim([-2, 6]);

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

figure('Name', 'HW');
subplot(1,1,1);
hold on;
shadeplot(zscoreMatrix(HW_Type == 1, :), 'SD', 'sem', 'LineWidth', sum(HW_Type == 1)/100);
shadeplot(zscoreMatrix(HW_Type == 2, :), 'SD', 'sem', 'LineWidth', sum(HW_Type == 2)/100);
shadeplot(zscoreMatrix(HW_Type == 3, :), 'SD', 'sem', 'LineWidth', sum(HW_Type == 3)/100);
ylim([-1.5, 3]);

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

figure('Name', 'HW-A');
subplot(1,1,1);
hold on;
shadeplot(zscoreMatrix(HW_A_Type == 1, :), 'SD', 'sem', 'LineWidth', sum(HW_A_Type == 1)/100);
shadeplot(zscoreMatrix(HW_A_Type == 2, :), 'SD', 'sem', 'LineWidth', sum(HW_A_Type == 2)/100);
shadeplot(zscoreMatrix(HW_A_Type == 3, :), 'SD', 'sem', 'LineWidth', sum(HW_A_Type == 3)/100);
ylim([-1.5, 3]);

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

figure('Name', 'HW-E');
subplot(1,1,1);
hold on;
shadeplot(zscoreMatrix(HW_E_Type == 1, :), 'SD', 'sem', 'LineWidth', sum(HW_E_Type == 1)/100);
shadeplot(zscoreMatrix(HW_E_Type == 2, :), 'SD', 'sem', 'LineWidth', sum(HW_E_Type == 2)/100);
shadeplot(zscoreMatrix(HW_E_Type == 3, :), 'SD', 'sem', 'LineWidth', sum(HW_E_Type == 3)/100);
ylim([-1.5, 3]);


%% Generate Table about sessions

SVMResult = load('D:\Data\Lobster\EventClassificationResult_4C\Output_AE_RFE_max_FI.mat');
BNBResult = load('D:\Data\Lobster\CV5_HEC_Result_BNB.mat');
BNB_LOO_Result = load('D:\Data\Lobster\HEC_Result_BNB.mat');
DistanceResultPath = 'D:\Data\Lobster\FineDistanceResult';

Sessions = table(...
    unique(Unit.Session),...
    zeros(40,3),...
    zeros(40,1),...
    zeros(40,1),...
    zeros(40,1),...
    zeros(40,1), zeros(40,1),...
    zeros(40,1), zeros(40,1), zeros(40,1),...
    zeros(40,1), zeros(40,1), zeros(40,1),...
    zeros(40,1), zeros(40,1), zeros(40,1),...
    zeros(40,1),...
    'VariableNames',...
        ["Session",...
        "SVM_HW_Accuracy",...
        "BNB_HW_Accuracy",...
        "BNB_HW_Accuracy_LOO",...
        "Location_Error",...
        "N_HE1", "N_HE2",...
        "N_HW1", "N_HW2", "N_HW3",...
        "N_HWA1", "N_HWA2", "N_HWA3",...
        "N_HWE1", "N_HWE2", "N_HWE3",...
        "NumCell"]...
    );

for i_session = 1 : 40
    session = Sessions.Session(i_session);
    
    Sessions.SVM_HW_Accuracy(i_session,:) = SVMResult.result{i_session}.balanced_accuracy_HW;
    Sessions.BNB_HW_Accuracy(i_session) = BNBResult.result{i_session}.balanced_accuracy_HWAE(2);
    Sessions.BNB_HW_Accuracy_LOO(i_session) = BNB_LOO_Result.result{i_session}.balanced_accuracy_HW_AE(2);
    
    LocationResult = load(fullfile(DistanceResultPath, strcat(session, 'result_distance.mat')), 'WholeTestResult');

    % L1 Error of True / L1 Error of the shuffled
    % 1 = No difference between original vs shuffled
    % 0.5 = Half of the error compared to the shuffled
    % 0 = Zero error
    Sessions.Location_Error(i_session) = ...
        mean(abs(LocationResult.WholeTestResult(:, 3) - LocationResult.WholeTestResult(:, 5)))...
        / mean(abs(LocationResult.WholeTestResult(:, 3) - LocationResult.WholeTestResult(:, 4))); 
    

    Sessions.N_HE1(i_session) = sum(Unit.Session == session &  HE_Type == 1);
    Sessions.N_HE2(i_session)= sum(Unit.Session == session &  HE_Type == 2);

    Sessions.N_HW1(i_session) = sum(Unit.Session == session &  HW_Type == 1);
    Sessions.N_HW2(i_session) = sum(Unit.Session == session &  HW_Type == 2);
    Sessions.N_HW3(i_session) = sum(Unit.Session == session &  HW_Type == 3);

    Sessions.N_HWA1(i_session) = sum(Unit.Session == session &  HW_A_Type == 1);
    Sessions.N_HWA2(i_session) = sum(Unit.Session == session &  HW_A_Type == 2);
    Sessions.N_HWA3(i_session) = sum(Unit.Session == session &  HW_A_Type == 3);

    Sessions.N_HWE1(i_session) = sum(Unit.Session == session &  HW_E_Type == 1);
    Sessions.N_HWE2(i_session) = sum(Unit.Session == session &  HW_E_Type == 2);
    Sessions.N_HWE3(i_session) = sum(Unit.Session == session &  HW_E_Type == 3);

    Sessions.NumCell(i_session) = sum(Unit.Session == session);
end

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
    linethickness = sum(CellType_ == clstr)/100;
    [~, l_, ~] = shadeplot(zscoreMatrix(CellType_ == clstr, :), 'SD', 'sem', 'LineWidth', linethickness);
    lines = [lines, l_];
    labels = [labels, sprintf('Type %d n=%d', clstr, sum(CellType_ == clstr))];
end
legend(lines, labels);

%% Draw Dendrograms
% 30 nodes only
figure(3);
clf;
dendrogram(Z_, 30, "Reorder", optimalleaforder(Z_, Y_, 'Criteria','group'), "ColorThreshold",mean(Z_(end-numCellCluster+1:end-numCellCluster+2,3)));

% All nodes
figure(4);
clf;
dendrogram(Z_, 632, "Reorder", optimalleaforder(Z_, Y_, 'Criteria','group'), "ColorThreshold",mean(Z_(end-numCellCluster+1:end-numCellCluster+2,3)));

% Animation
figure(5);
clf;
lines = dendrogram(Z_, 632, "Reorder", optimalleaforder(Z_, Y_, 'Criteria','group'));
for i = 1 : 631
    lines(i).Color = 'w';
end
for i = 1 : 631
    lines(i).Color = 'k';
    drawnow;
    pause(0.01);
end

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

