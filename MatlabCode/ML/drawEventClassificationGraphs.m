%% drawEventClassificationGraphs
% Analyse event classification and draw graphs
loadxkcd;
load('HEC_Result.mat'); %% HEC Result Data
sessionNames = string(sessionNames);
tankNames = string(tankNames);

%% Classifier accuracy
prismOut_HEHW = table(zeros(40,1), zeros(40,1), zeros(40,1), 'VariableNames', ["Shuffled", "Predicted", "Predicted_Max"]);

for session = 1 : 40
    prismOut_HEHW.Shuffled(session) = result{session}.balanced_accuracy_HEHW(1);
    prismOut_HEHW.Predicted(session) = result{session}.balanced_accuracy_HEHW(2);
    prismOut_HEHW.Predicted_Max(session) = result{session}.balanced_accuracy_HEHW(3);
end

prismOut_HE_AE = table(zeros(40,1), zeros(40,1), zeros(40,1), 'VariableNames', ["Shuffled", "Predicted", "Predicted_Max"]);

for session = 1 : 40
    prismOut_HE_AE.Shuffled(session) = result{session}.balanced_accuracy_HE_AE(1);
    prismOut_HE_AE.Predicted(session) = result{session}.balanced_accuracy_HE_AE(2);
    prismOut_HE_AE.Predicted_Max(session) = result{session}.balanced_accuracy_HE_AE(3);
end

prismOut_HW_AE = table(zeros(40,1), zeros(40,1), zeros(40,1), 'VariableNames', ["Shuffled", "Predicted", "Predicted_Max"]);

for session = 1 : 40
    prismOut_HW_AE.Shuffled(session) = result{session}.balanced_accuracy_HW_AE(1);
    prismOut_HW_AE.Predicted(session) = result{session}.balanced_accuracy_HW_AE(2);
    prismOut_HW_AE.Predicted_Max(session) = result{session}.balanced_accuracy_HW_AE(3);
end



%% Classifier Accuracy Confusion Matrix

% third dimension : 1:PL 2 : IL
confusionMatrix_ctrl = zeros(4,4,2); % shuffled
confusionMatrix_real = zeros(4,4,2);

for session = 1 : 40
    % Get Brain area index
    if contains(tankNames(session), "PL")
        brainAreaIdx = 1;
    else
        brainAreaIdx = 2;
    end

    for s = 1 : size(result{session}.HEHW_prediction,1) / 2
        %% Input control data
        % input Head Entry Data
        rowIdx = result{session}.HE_AE_prediction(s,1); % actual
        % if HEHW prediction is equal, then nothing happens.
        % if HEHW preidction is not equal, add 2 ==> assigned to HW's A or E
        colIdx = +2*(result{session}.HEHW_prediction(s,2) ~= result{session}.HEHW_prediction(s,1)) ... 
            + result{session}.HE_AE_prediction(s,2);
        confusionMatrix_ctrl(rowIdx, colIdx, brainAreaIdx) = confusionMatrix_ctrl(rowIdx, colIdx, brainAreaIdx) + 1;

        r = s + size(result{session}.HEHW_prediction,1) / 2;
        % input Head Withdrawal Data
        rowIdx = result{session}.HW_AE_prediction(s,1); % actual
        % if HEHW prediction is equal, then nothing happens.
        % if HEHW preidction is not equal, subtract 2 ==> assigned to HE's A or E
        colIdx = -2*(result{session}.HEHW_prediction(r,2) ~= result{session}.HEHW_prediction(r,1)) ... 
            + result{session}.HW_AE_prediction(s,2);
        confusionMatrix_ctrl(rowIdx, colIdx, brainAreaIdx) = confusionMatrix_ctrl(rowIdx, colIdx, brainAreaIdx) + 1;
        
        %% Input real data
        % input Head Entry Data
        rowIdx = result{session}.HE_AE_prediction(s,1); % actual
        % if HEHW prediction is equal, then nothing happens.
        % if HEHW preidction is not equal, add 2 ==> assigned to HW's A or E
        colIdx = +2*(result{session}.HEHW_prediction(s,3) ~= result{session}.HEHW_prediction(s,1)) ... 
            + result{session}.HE_AE_prediction(s,3);
        confusionMatrix_real(rowIdx, colIdx, brainAreaIdx) = confusionMatrix_real(rowIdx, colIdx, brainAreaIdx) + 1;

        r = s + size(result{session}.HEHW_prediction,1) / 2;
        % input Head Withdrawal Data
        rowIdx = result{session}.HW_AE_prediction(s,1); % actual
        % if HEHW prediction is equal, then nothing happens.
        % if HEHW preidction is not equal, subtract 2 ==> assigned to HE's A or E
        colIdx = -2*(result{session}.HEHW_prediction(r,3) ~= result{session}.HEHW_prediction(r,1)) ... 
            + result{session}.HW_AE_prediction(s,3);
        confusionMatrix_real(rowIdx, colIdx, brainAreaIdx) = confusionMatrix_real(rowIdx, colIdx, brainAreaIdx) + 1;
    end
end

%% Draw confusion matrix for all session
fig = figure('Name', 'Confusion Marix : All');

cmap_All = [...
    linspace(1, xkcd.purple(1), 100)',...
    linspace(1, xkcd.purple(2), 100)',...
    linspace(1, xkcd.purple(3), 100)'];

cMat_ctrl_ = sum(confusionMatrix_ctrl, 3);
cMat_real_ = sum(confusionMatrix_real, 3);
cMat_ctrl_ = cMat_ctrl_ ./ repmat(sum(cMat_ctrl_,2), 1,4);
cMat_real_ = cMat_real_ ./ repmat(sum(cMat_real_,2), 1,4);

subplot(1,2,1);
ax1 = heatmap({'AHE', 'EHE', 'AHW', 'EHW'}, {'AHE', 'EHE', 'AHW', 'EHW'}, cMat_ctrl_);
caxis([0, 1]);
colormap(ax1, cmap_All);
ax1.CellLabelFormat = '%0.2f';
xlabel('Decoded');
ylabel('True');
title('Shuffled');

subplot(1,2,2);
ax2 = heatmap({'AHE', 'EHE', 'AHW', 'EHW'}, {'AHE', 'EHE', 'AHW', 'EHW'}, cMat_real_);
caxis([0, 1]);
colormap(ax2, cmap_All);
ax2.CellLabelFormat = '%0.2f';
xlabel('Decoded');
ylabel('True');
title('Original');

%% Draw confusion matrix for each brain area
fig = figure('Name', 'Confusion Matrix : PL IL');

cmap_PL = [...
    linspace(1, xkcd.pig_pink(1), 100)',...
    linspace(1, xkcd.pig_pink(2), 100)',...
    linspace(1, xkcd.pig_pink(3), 100)'];
cmap_IL = [...
    linspace(1, xkcd.sky_blue(1), 100)',...
    linspace(1, xkcd.sky_blue(2), 100)',...
    linspace(1, xkcd.sky_blue(3), 100)'];

% PL
cMat_ctrl_ = confusionMatrix_ctrl(:,:,1);
cMat_real_ = confusionMatrix_real(:,:,1);
cMat_ctrl_ = cMat_ctrl_ ./ repmat(sum(cMat_ctrl_,2), 1,4);
cMat_real_ = cMat_real_ ./ repmat(sum(cMat_real_,2), 1,4);

subplot(2,2,1);
ax1 = heatmap({'AHE', 'EHE', 'AHW', 'EHW'}, {'AHE', 'EHE', 'AHW', 'EHW'}, cMat_ctrl_);
caxis([0, 1]);
colormap(ax1, cmap_PL);
ax1.CellLabelFormat = '%0.2f';
xlabel('Decoded');
ylabel('True');
title('Shuffled');

subplot(2,2,2);
ax2 = heatmap({'AHE', 'EHE', 'AHW', 'EHW'}, {'AHE', 'EHE', 'AHW', 'EHW'}, cMat_real_);
caxis([0, 1]);
colormap(ax2, cmap_PL);
ax2.CellLabelFormat = '%0.2f';
xlabel('Decoded');
ylabel('True');
title('Original');

% IL
cMat_ctrl_ = confusionMatrix_ctrl(:,:,2);
cMat_real_ = confusionMatrix_real(:,:,2);
cMat_ctrl_ = cMat_ctrl_ ./ repmat(sum(cMat_ctrl_,2), 1,4);
cMat_real_ = cMat_real_ ./ repmat(sum(cMat_real_,2), 1,4);

subplot(2,2,3);
ax3 = heatmap({'AHE', 'EHE', 'AHW', 'EHW'}, {'AHE', 'EHE', 'AHW', 'EHW'}, cMat_ctrl_);
caxis([0, 1]);
colormap(ax3, cmap_IL);
ax3.CellLabelFormat = '%0.2f';
xlabel('Decoded');
ylabel('True');
title('Shuffled');

subplot(2,2,4);
ax4 = heatmap({'AHE', 'EHE', 'AHW', 'EHW'}, {'AHE', 'EHE', 'AHW', 'EHW'}, cMat_real_);
caxis([0, 1]);
ax4.CellLabelFormat = '%0.2f';
colormap(ax4, cmap_IL);
xlabel('Decoded');
ylabel('True');
title('Original');

set(gca, 'FontName', 'Noto Sans');

%%

% %% Generate Feature Importance Score data
% % This data is plotted in Prism
% % `load("ClassifiedUnitData.mat")` must be loaded first.
% idx = 1;
% FI = zeros(size(Units, 1), 1);
% FI_AHE = zeros(size(Units,1),1);
% FI_EHE = zeros(size(Units,1),1);
% FI_AHW = zeros(size(Units,1),1);
% FI_EHW = zeros(size(Units,1),1);
% 
% for session = 1 : 40
%     numUnitInSession = size(result{session}.importance_score,2);
%     FI(idx : idx + numUnitInSession - 1) = mean(result{session}.importance_score, 1)';
%     FI_AHE(idx : idx + numUnitInSession - 1) = mean(result{session}.importance_score_AHE, 1)';
%     FI_EHE(idx : idx + numUnitInSession - 1) = mean(result{session}.importance_score_EHE, 1)';
%     FI_AHW(idx : idx + numUnitInSession - 1) = mean(result{session}.importance_score_AHW, 1)';
%     FI_EHW(idx : idx + numUnitInSession - 1) = mean(result{session}.importance_score_EHW, 1)';
%     idx = idx + numUnitInSession;
% end
% 
% Units = [Units, table(FI, FI_AHE, FI_EHE, FI_AHW, FI_EHW, 'VariableNames', ["FI", "FI_AHE", "FI_EHE", "FI_AHW", "FI_EHW"])];
% 
% % FI on total accuracy
% FI_by_AHE_class.pre = Units.FI(Units.first_LICK_A_type == 1) * 100;
% FI_by_AHE_class.peri = Units.FI(Units.first_LICK_A_type == 2) * 100;
% FI_by_AHE_class.post = Units.FI(Units.first_LICK_A_type == 3) * 100;
% FI_by_AHE_class.none = Units.FI(Units.first_LICK_A_type == 0) * 100;
% 
% FI_by_EHE_class.pre = Units.FI(Units.first_LICK_E_type == 1) * 100;
% FI_by_EHE_class.peri = Units.FI(Units.first_LICK_E_type == 2) * 100;
% FI_by_EHE_class.post = Units.FI(Units.first_LICK_E_type == 3) * 100;
% FI_by_EHE_class.none = Units.FI(Units.first_LICK_E_type == 0) * 100;
% 
% FI_by_AHW_class.pre = Units.FI(Units.valid_IROF_A_type == 1) * 100;
% FI_by_AHW_class.peri = Units.FI(Units.valid_IROF_A_type == 2) * 100;
% FI_by_AHW_class.post = Units.FI(Units.valid_IROF_A_type == 3) * 100;
% FI_by_AHW_class.none = Units.FI(Units.valid_IROF_A_type == 0) * 100;
% 
% FI_by_EHW_class.pre = Units.FI(Units.valid_IROF_E_type == 1) * 100;
% FI_by_EHW_class.peri = Units.FI(Units.valid_IROF_E_type == 2) * 100;
% FI_by_EHW_class.post = Units.FI(Units.valid_IROF_E_type == 3) * 100;
% FI_by_EHW_class.none = Units.FI(Units.valid_IROF_E_type == 0) * 100;
% 
% % FI on each class accuracy
% FI_AHE_by_AHE_class.pre = Units.FI_AHE(Units.first_LICK_A_type == 1) * 100;
% FI_AHE_by_AHE_class.peri = Units.FI_AHE(Units.first_LICK_A_type == 2) * 100;
% FI_AHE_by_AHE_class.post = Units.FI_AHE(Units.first_LICK_A_type == 3) * 100;
% FI_AHE_by_AHE_class.none = Units.FI_AHE(Units.first_LICK_A_type == 0) * 100;
% 
% FI_EHE_by_EHE_class.pre = Units.FI_EHE(Units.first_LICK_E_type == 1) * 100;
% FI_EHE_by_EHE_class.peri = Units.FI_EHE(Units.first_LICK_E_type == 2) * 100;
% FI_EHE_by_EHE_class.post = Units.FI_EHE(Units.first_LICK_E_type == 3) * 100;
% FI_EHE_by_EHE_class.none = Units.FI_EHE(Units.first_LICK_E_type == 0) * 100;
% 
% FI_AHW_by_AHW_class.pre = Units.FI_AHW(Units.valid_IROF_A_type == 1) * 100;
% FI_AHW_by_AHW_class.peri = Units.FI_AHW(Units.valid_IROF_A_type == 2) * 100;
% FI_AHW_by_AHW_class.post = Units.FI_AHW(Units.valid_IROF_A_type == 3) * 100;
% FI_AHW_by_AHW_class.none = Units.FI_AHW(Units.valid_IROF_A_type == 0) * 100;
% 
% FI_EHW_by_EHW_class.pre = Units.FI_EHW(Units.valid_IROF_E_type == 1) * 100;
% FI_EHW_by_EHW_class.peri = Units.FI_EHW(Units.valid_IROF_E_type == 2) * 100;
% FI_EHW_by_EHW_class.post = Units.FI_EHW(Units.valid_IROF_E_type == 3) * 100;
% FI_EHW_by_EHW_class.none = Units.FI_EHW(Units.valid_IROF_E_type == 0) * 100;
% 
% 
% %% Load Number of Units vs Accuracy Data
% 
% % numSession = 40;
% % maxUnit = 26;
% % numRepeat = 5;
% % unitAccuracy = cell(numSession, 2, numRepeat, maxUnit);
% % 
% % for session = 1 : numel(sessionPaths)
% %     TANK_name = cell2mat(sessionPaths{session});
% %     TANK_location = char(strcat(basePath, filesep, TANK_name));
% %     % Load and process data
% %     load(TANK_location);
% %     
% %     % Result (shuffle/real) x (repeat) x (unit)
% %     unitAccuracy(session, :, :, 1 : size(result, 3)) = num2cell(result);    
% % end
% 
% %% Draw Number of Units vs Accuracy Graph
% 
% figure('Position', [680   553   443   425]);
% subplot(1,1,1);
% hold on;
% 
% for session = 1 : numSession
%     plot(mean(cell2mat(squeeze(unitAccuracy(session, 1, :, :))), 1), 'Color', xkcd.light_grey);
%     plot(mean(cell2mat(squeeze(unitAccuracy(session, 2, :, :))), 1), 'Color', xkcd.light_grey);
% end
% 
% meanAccuracy = zeros(2, 26);
% for i = 1 : 26
%     meanAccuracy(1,i) = mean(cell2mat(squeeze(unitAccuracy(:, 1, :, i))), 'all');
%     meanAccuracy(2,i) = mean(cell2mat(squeeze(unitAccuracy(:, 2, :, i))), 'all');
% end
% 
% l1 = plot(meanAccuracy(1,:), 'Color', 'k', 'LineWidth', 2, 'LineStyle', '--');
% l2 = plot(meanAccuracy(2,:), 'Color', 'k', 'LineWidth', 2);
% 
% xlabel('Number of Units', 'FontName', 'Noto Sans');
% ylabel('Balanced accuracy');
% 
% xlim([1, 26]);
% ylim([0.2, 1]);
% 
% legend([l1, l2], {'Shuffled', 'Real'});
% 
% set(gca, 'FontName', 'Noto Sans');
