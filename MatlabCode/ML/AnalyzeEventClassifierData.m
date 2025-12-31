%% AnalyzeEventClassifierData
% Analyse event classification and draw graphs
%loadxkcd;
load('D:\Data\Lobster\BNB_Result_unitshffle.mat'); %% HEC Result Data
sessionNames = string(sessionNames);
tankNames = string(tankNames);

%% Classifier accuracy
prismOut_accuracy = table(zeros(40,1), zeros(40,1), 'VariableNames', ["Shuffled", "Predicted"]);

for session = 1 : 40
    prismOut_accuracy.Shuffled(session) = result{session}.balanced_accuracy_HEHW(1);
    prismOut_accuracy.Predicted(session) = result{session}.balanced_accuracy_HEHW(2);
end

prismOut_HEAE = table(zeros(40,1), zeros(40,1), 'VariableNames', ["Shuffled", "Predicted"]);

for session = 1 : 40
    prismOut_HEAE.Shuffled(session) = result{session}.balanced_accuracy_HEAE(1);
    prismOut_HEAE.Predicted(session) = result{session}.balanced_accuracy_HEAE(2);
end

prismOut_HWAE = table(zeros(40,1), zeros(40,1), 'VariableNames', ["Shuffled", "Predicted"]);

for session = 1 : 40
    prismOut_HWAE.Shuffled(session) = result{session}.balanced_accuracy_HWAE(1);
    prismOut_HWAE.Predicted(session) = result{session}.balanced_accuracy_HWAE(2);
end

%% Classifier accuracy (Whole = Hierarchical)
prismOut_accuracy = table(zeros(40,1), zeros(40,1), 'VariableNames', ["Shuffled", "Predicted"]);
prismOut_balancedAccuracy = table(zeros(40,1), zeros(40,1), 'VariableNames', ["Shuffled", "Predicted"]);

for session = 1 : 40
    HEHW_result = result{session}.WholeTestResult_HEHW >= 0.5;
    HEAE_result = result{session}.WholeTestResult_HEAE >= 0.5;
    HWAE_result = result{session}.WholeTestResult_HWAE >= 0.5;
    numTrial = size(result{session}.WholeTestResult_HEAE,1);

    tot_result = HEHW_result .* 2 + [HEAE_result; HWAE_result] + 1;
    
    % accuracy
    prismOut_accuracy.Shuffled(session) = ...
        sum(tot_result(:,1) == tot_result(:,2));
    prismOut_accuracy.Predicted(session) = ...
        sum(tot_result(:,1) == tot_result(:,3));

    % balanced accuracy
    prismOut_balancedAccuracy.Shuffled(session) = ...
        1/4*(...
        sum(tot_result(tot_result(:,1) == 1, 2) == 1) / sum(tot_result(:,1) == 1) +...
        sum(tot_result(tot_result(:,1) == 2, 2) == 2) / sum(tot_result(:,1) == 2) +...
        sum(tot_result(tot_result(:,1) == 3, 2) == 3) / sum(tot_result(:,1) == 3) +...
        sum(tot_result(tot_result(:,1) == 4, 2) == 4) / sum(tot_result(:,1) == 4));
    prismOut_balancedAccuracy.Predicted(session) = ...
        1/4*(...
        sum(tot_result(tot_result(:,1) == 1, 3) == 1) / sum(tot_result(:,1) == 1) +...
        sum(tot_result(tot_result(:,1) == 2, 3) == 2) / sum(tot_result(:,1) == 2) +...
        sum(tot_result(tot_result(:,1) == 3, 3) == 3) / sum(tot_result(:,1) == 3) +...
        sum(tot_result(tot_result(:,1) == 4, 3) == 4) / sum(tot_result(:,1) == 4));
end

%% Classifier Accuracy Confusion Matrix
% third dimension : 1:PL 2 : IL
confusionMatrix_ctrl = zeros(4,4,2); % shuffled
confusionMatrix_real = zeros(4,4,2);

% Row indicate Actual value, Column indicate Predicted value
% HE-A | HE-E | HW-A | HW-E

for session = 1 : 40
    % Get Brain area index
    if contains(tankNames(session), "PL")
        brainAreaIdx = 1;
    else
        brainAreaIdx = 2;
    end

    numTrial = size(result{session}.WholeTestResult_HEAE,1);

    HEHW_result = result{session}.WholeTestResult_HEHW >= 0.5;
    HEAE_result = result{session}.WholeTestResult_HEAE >= 0.5;
    HWAE_result = result{session}.WholeTestResult_HWAE >= 0.5;

    for trial = 1 : numTrial
        %% Input control data

        % input Head Entry Data
        rowIdx = HEAE_result(trial,1) + 1; % 1: HE-A, 2: HE-E
        % if successfully classified HE as HE, then nothing happens.
        % but if classified HE as HW, add 2 ==> assigned to HW's A or E
        colIdx = 2*(HEHW_result(trial,1) ~= HEHW_result(trial,2)) ... 
            + HEAE_result(trial,2) + 1;
        confusionMatrix_ctrl(rowIdx, colIdx, brainAreaIdx) = confusionMatrix_ctrl(rowIdx, colIdx, brainAreaIdx) + 1;

        % input Head Withdrawal Data
        rowIdx = HWAE_result(trial,1) + 3; % 3: HW-A, 4: HW-E
        % if successfully classified HW as HW, then nothing happens.
        % but if classified HW as HE, subtract 2 ==> assigned to HE's A or E
        colIdx = -2*(HEHW_result(trial + numTrial,1) ~= HEHW_result(trial + numTrial,2)) ... 
            + HWAE_result(trial,2) + 3;
        confusionMatrix_ctrl(rowIdx, colIdx, brainAreaIdx) = confusionMatrix_ctrl(rowIdx, colIdx, brainAreaIdx) + 1;
        
        %% Input real data
        % input Head Entry Data
        rowIdx = HEAE_result(trial,1) + 1; % 1: HE-A, 2: HE-E
        % if successfully classified HE as HE, then nothing happens.
        % but if classified HE as HW, add 2 ==> assigned to HW's A or E
        colIdx = 2*(HEHW_result(trial,1) ~= HEHW_result(trial,3)) ... 
            + HEAE_result(trial,3) + 1;
        confusionMatrix_real(rowIdx, colIdx, brainAreaIdx) = confusionMatrix_real(rowIdx, colIdx, brainAreaIdx) + 1;

        % input Head Withdrawal Data
        rowIdx = HWAE_result(trial,1) + 3; % 3: HW-A, 4: HW-E
        % if successfully classified HW as HW, then nothing happens.
        % but if classified HW as HE, subtract 2 ==> assigned to HE's A or E
        colIdx = -2*(HEHW_result(trial + numTrial,1) ~= HEHW_result(trial + numTrial,3)) ... 
            + HWAE_result(trial,3) + 3;
        confusionMatrix_real(rowIdx, colIdx, brainAreaIdx) = confusionMatrix_real(rowIdx, colIdx, brainAreaIdx) + 1;
    end
end

%% Draw confusion matrix for all session
fig = figure('Name', 'Confusion Marix : All');

cmap_All = [...
    linspace(1, 241/255, 100)',...
    linspace(1, 136/255, 100)',...
    linspace(1, 26/255, 100)'];

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
set(gca, 'FontName', 'Noto Sans');

subplot(1,2,2);
ax2 = heatmap({'AHE', 'EHE', 'AHW', 'EHW'}, {'AHE', 'EHE', 'AHW', 'EHW'}, cMat_real_);
caxis([0, 1]);
colormap(ax2, cmap_All);
ax2.CellLabelFormat = '%0.2f';
xlabel('Decoded');
ylabel('True');
title('Original');
set(gca, 'FontName', 'Noto Sans');
set(gcf, 'Position', [428   505   555   215]);

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

%% Check if Head Withdrawal time as correlation with Accuracy
correct_HW_Time = [];
wrong_HW_Time = [];
for session = 1 : 40
    % Get HW Time
    behav_data = load(fullfile("D:\Data\Lobster\BehaviorData", strcat(sessionNames(session, :), '.mat')));
    ParsedData = behav_data.ParsedData;
    HW_Time = zeros(size(ParsedData,1), 1);
    for trial = 1 : size(ParsedData,1)
        attackTime = ParsedData{trial, 4}(1);
        nearAttackIRindex = find(ParsedData{trial, 2}(:,1) < attackTime,1,'last');
        HW_Time(trial) = ParsedData{trial, 2}(nearAttackIRindex, 2) - ParsedData{trial, 2}(1, 1);
    end

    classifier_result = result{session}.WholeTestResult_HWAE >= 0.5;
    classifier_result = classifier_result(:,1) == classifier_result(:,3);

    % Consider only 6s trials and AW
    for trial = 1 : size(ParsedData,1)
        % 6-sec trials
        if ParsedData{trial, 4}(1) - ParsedData{trial, 2}(1) > 4.5
            if HW_Time(trial) < 6  % AW
                if classifier_result(trial)
                    correct_HW_Time = [correct_HW_Time, HW_Time(trial)];
                else
                    wrong_HW_Time = [wrong_HW_Time, HW_Time(trial)];
                end
            end
        end
    end
end
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
