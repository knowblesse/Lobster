%% ConcatenateUnitData
ClassifyUnits;

clearvars -except Unit
px2cm = 0.169;

%% Feature Importance - Fine Distance Regressor
resultPath = 'D:\Data\Lobster\FineDistanceResult_syncFixed_rmEncounter';

filelist = dir(resultPath);
sessionPaths = regexp({filelist.name},'^#\S*.mat','match');
sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));

FI_Distance = zeros(size(Unit,1),1);
FI_Distance_5bin = zeros(size(Unit,1), 5);

fprintf('session : 00/40');
for session = 1 : 40
    sessionName = cell2mat(regexp(cell2mat(sessionPaths{session}), '^#.*?L', 'match'));
    MAT_filename = cell2mat(sessionPaths{session});
    MAT_filePath = char(strcat(resultPath, filesep, MAT_filename));
    load(MAT_filePath); % PFITestResult, WholeTestResult(row, col, true d , shuffled d, pred d)
    
    % calculate bin index of the column
    col_index = discretize(WholeTestResult(:,2), 5);
    err_original_5bin = zeros(1,5);
    for bin = 1 : 5
        err_original_5bin(bin) = mean(abs(WholeTestResult(col_index == bin,3) - WholeTestResult(col_index == bin,5))) * px2cm;
    end

    % calculate FI for every units
    err_original = mean(abs(WholeTestResult(:,3) - WholeTestResult(:,5))) * px2cm;
    err_shuffled = mean(abs(WholeTestResult(:,3) - WholeTestResult(:,4))) * px2cm;
    for unit = 1 : size(PFITestResult, 2)
        err_corrupted = mean(squeeze(abs(WholeTestResult(:,3) - PFITestResult(:,unit, :))), 2) * px2cm;

        % FI Factor Difference
        FI_Distance(Unit.Session == sessionName & Unit.Cell == unit) = ...
            mean(err_corrupted) - err_original;

        % FI 5bin
        err_corrupted_5bin = zeros(1,5);
        for bin = 1 : 5
            err_corrupted_5bin(bin) = mean(err_corrupted(col_index == bin));
        end
        FI_Distance_5bin(Unit.Session == sessionName & Unit.Cell == unit, :) = err_corrupted_5bin;

    end
    fprintf('\b\b\b\b\b%02d/40', session);
end

Unit = [Unit, table(FI_Distance, FI_Distance_5bin, 'VariableNames', {'FI_Distance', 'FI_Distance_5bin'})];

%% Feature Importance - Event Classifier
resultPath = 'D:\Data\Lobster\BNB_Result_fullshuffle.mat';

load(resultPath);

sessionNames = string(sessionNames);

EC_Score = []; % score of the session
FI_EC_ACC = []; % FI using accuracy difference
FI_EC_CE = []; % FI using cross entropy difference
FI_EC_FP = []; % FI using feature probabilities

for session = 1 : 40
    fprintf('%s\n', sessionNames{session});
    numBin = 40;
    numCell = numel(result{session}.PFICrossEntropy_HEAE);

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
end

Unit = [Unit, table(EC_Score, FI_EC_ACC, FI_EC_CE, FI_EC_FP, 'VariableNames', {'EC_Score', 'FI_EC_ACC','FI_EC_CE','FI_EC_FP'})];
clearvars -except Unit

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
