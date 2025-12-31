%% FeatureImporatanceMainScript
% Main script for feature important analysis

%% Create `Unit` table which contains all information for each unit
ClassifyUnits;

clearvars -except Unit
px2cm = 0.169;


%% Feature Importance - Fine Distance Regressor
% For main analysis, I have use `sync_Fixed_June`.
% However for analysis involving PFI, use `sync_Fixed_May`
resultPath = 'D:\Data\Lobster\FineDistanceResult_syncFixed_May';

filelist = dir(resultPath);
sessionPaths = regexp({filelist.name},'^#\S*.mat','match');
sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));

FI_Distance = zeros(size(Unit,1),1);

anova_result = cell(40,1);

fprintf('session : 00/40');
for session = 1 : 40
    sessionName = cell2mat(regexp(cell2mat(sessionPaths{session}), '^#.*?L', 'match'));
    MAT_filename = cell2mat(sessionPaths{session});
    MAT_filePath = char(strcat(resultPath, filesep, MAT_filename));
    load(MAT_filePath); % PFITestResult, WholeTestResult(row, col, true d , shuffled d, pred d)

    % calculate FI for every units
    err_original = mean(abs(WholeTestResult(:,3) - WholeTestResult(:,5))) * px2cm;
    err_shuffled = mean(abs(WholeTestResult(:,3) - WholeTestResult(:,4))) * px2cm;
    numUnit = size(PFITestResult, 2);
    for unit = 1 : numUnit
        err_corrupted = mean(squeeze(abs(WholeTestResult(:,3) - PFITestResult(:,unit, :))), 2) * px2cm;

        % FI Factor Difference
        FI_Distance(Unit.Session == sessionName & Unit.Cell == unit) = ...
            mean(err_corrupted) - err_original;
    end
    
    % calculate PFI result
    %   calculate mean error (average across all datapoints) for each shuffle.
    %   get 2D (shuffles x neuron) data for ANOVA
    anova_data = squeeze(mean(abs(WholeTestResult(:,3) - PFITestResult), 1))';  
    
    anova_result{session} = anova1(anova_data, 1:numUnit, 'off');
    fprintf('\b\b\b\b\b%02d/40', session);
end

Unit = [Unit, table(FI_Distance, 'VariableNames', {'FI_Distance'})];

%% Draw histogram of all neurons feature importance
fig = figure(1);
set(fig, 'Position', [200, 200, 300, 300]);
clf;
histogram(Unit.FI_Distance, 100, 'FaceColor', [0.3, 0.3, 0.3], 'LineStyle', 'none');
xlabel('Error (cm)');
ylabel('Count (neuron)');
fprintf("\n50%% Value : %.2f\n", prctile(Unit.FI_Distance, 50));
p = prctile(Unit.FI_Distance, 95);
line([p, p], [0, 100], 'Color', 'r');

xlim([0, 5]);
ylim([0, 100]);

yticks(0:20:100);

set(gca, 'FontName', 'Noto Sans')

%% Feature Importance - Event Classifier
resultPath = 'D:\Data\Lobster\BNB_Result_fullshuffle.mat';

load(resultPath);

sessionNames = string(sessionNames);

EC_Score = []; % score of the session
FI_EC_ACC = []; % FI using accuracy difference
FI_EC_CE = []; % FI using cross entropy difference
FI_EC_FP = []; % FI using feature probabilities

anova_result_ec = cell(40,1);
for session = 1 : 40
    fprintf('%s\n', sessionNames{session});
    numBin = 40;
    numCell = numel(result{session}.PFICrossEntropy_HEAE);
    numUnit = size(result{session}.PFITestResult_HWAE, 2);
    numRepeat = 10;

    EC_Score = [EC_Score;...
        repmat([...
            result{session}.balanced_accuracy_HEHW,...
            result{session}.balanced_accuracy_HEAE,...
            result{session}.balanced_accuracy_HWAE],...
            numCell, 1)...
        ];
    
    FI_EC_ACC = [FI_EC_ACC;...
        permutation_feature_importance(result{session}.WholeTestResult_HWAE >= 0.5, result{session}.PFITestResult_HWAE >= 0.5, 'method','difference')];

    FI_EC_CE = [FI_EC_CE;...
        result{session}.PFICrossEntropy_HWAE];
    
    t = reshape(...
                    permute(result{session}.feature_prob_HWAE, [3,2,1]),... % 
                    numBin, numCell, 2, 5);

    logFeatureProb = mean(...
        squeeze(...
            sum(...
                t(:,:,:,:)...
                ,1)... % sum of all log(Prob) from all units. => joint Prob.
            )...
        ,3);
    FI_EC_FP = [FI_EC_FP; oddRatio(logFeatureProb(:,1), logFeatureProb(:,2))];
    
    % calculate PFI result
    %   calculate mean error (average across all datapoints) for each shuffle.
    %   get 2D (shuffles x neuron) data for ANOVA
    val_true = result{session}.WholeTestResult_HWAE(:, 1);
    
    epsilon = 1e-10;
    val_pred = max(epsilon, min(1-epsilon, result{session}.PFITestResult_HWAE >= 0.5));

    % Calculate the cross-entropy
    crossEntropy = -mean(val_true .* log(val_pred) + (1 - val_true) .* log(1 - val_pred));
    %anova_data = squeeze(crossEntropy)';
    % Accuracy
    
    pfi = zeros(numRepeat, numUnit);
    for unit = 1 : numUnit
        for rep = 1 : numRepeat
            pfi(rep, unit) = result{session}.balanced_accuracy_HWAE(2) - ...
                balanced_accuracy_score(...
                    result{session}.WholeTestResult_HWAE(:,1),...
                    result{session}.PFITestResult_HWAE(:, unit, rep) >= 0.5...
                    );
        end
    end
    
    %anova_result_ec{session} = anova1(anova_data, 1:numUnit, 'off');
    anova_result_ec{session} = anova1(pfi(1:5, :), 1:numUnit, 'off');
end

Unit = [Unit, table(EC_Score, FI_EC_ACC, FI_EC_CE, FI_EC_FP, 'VariableNames', {'EC_Score', 'FI_EC_ACC','FI_EC_CE','FI_EC_FP'})];
%clearvars -except Unit

%% Draw histogram of all neurons feature importance
fig = figure(2);
set(fig, 'Position', [200, 200, 300, 300]);
clf;
histogram(Unit.FI_EC_ACC, 100, 'FaceColor', [0.3, 0.3, 0.3], 'LineStyle', 'none');
xlabel('Accuracy Drop');
ylabel('Count (neuron)');

p = prctile(Unit.FI_EC_ACC, 95);
line([p, p], [0, 140], 'Color', 'r');

xlim([-0.1, 0.1]);
ylim([0, 140]);

yticks(0:20:140);

set(gca, 'FontName', 'Noto Sans')

%% Load without ace data and compare




%% 
Unit.FI_EC_FP(Unit.Group_HE == 1)
Unit.FI_EC_FP(Unit.Group_HE == 2)
Unit.FI_EC_FP(Unit.Group_HW == 1)
Unit.FI_EC_FP(Unit.Group_HW == 2)
Unit.FI_EC_FP(Unit.Group_HW == 3)

Unit.FI_EC_FP(Unit.Group_HE == 1 & Unit.Group_HW == 1)
Unit.FI_EC_FP(Unit.Group_HE == 2 & Unit.Group_HW == 2)
Unit.FI_EC_FP(~(...
    (Unit.Group_HE == 1 & Unit.Group_HW == 1) | ...
    (Unit.Group_HE == 2 & Unit.Group_HW == 2)))


Unit.FI_Distance(Unit.Group_HE == 1)
Unit.FI_Distance(Unit.Group_HE == 2)
Unit.FI_Distance(Unit.Group_HW == 1)
Unit.FI_Distance(Unit.Group_HW == 2)
Unit.FI_Distance(Unit.Group_HW == 3)

Unit.FI_Distance(Unit.Group_HE == 1 & Unit.Group_HW == 1)
Unit.FI_Distance(Unit.Group_HE == 2 & Unit.Group_HW == 2)
Unit.FI_Distance(~(...
    (Unit.Group_HE == 1 & Unit.Group_HW == 1) | ...
    (Unit.Group_HE == 2 & Unit.Group_HW == 2)))

Unit.FI_Distance_5bin(Unit.Group_HE == 1 & Unit.Group_HW == 1, :)
Unit.FI_Distance_5bin(Unit.Group_HE == 2 & Unit.Group_HW == 2, :)
Unit.FI_Distance_5bin(~(...
    (Unit.Group_HE == 1 & Unit.Group_HW == 1) | ...
    (Unit.Group_HE == 2 & Unit.Group_HW == 2)), :)


%% 
[h, p] = ttest2(...
    abs(Unit.FI_EC_FP((Unit.Group_HW_A == 0) & (Unit.EC_Score(:,6) > 0.6))),...
    abs(Unit.FI_EC_FP((Unit.Group_HW_A == 3) & (Unit.EC_Score(:,6) > 0.6))))

%% Check if there is a correlation between FI_Distance and 
sessionNames = unique(Unit.Session);
corrResult = zeros(40,1);
for session = 1 : 40
    sessionName = sessionNames(session);
    FI_distance = Unit.FI_Distance(Unit.Session == sessionName);
    FI_event = Unit.FI_EC_CE(Unit.Session == sessionName);
    corrResult(session) = corr(FI_distance, FI_event);
end



