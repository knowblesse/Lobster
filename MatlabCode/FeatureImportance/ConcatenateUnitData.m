%% ConcatenateUnitData
ClassifyUnits;

clearvars -except Unit

%% Feature Importance - Fine Distance Regressor
resultPath = 'D:\Data\Lobster\FineDistanceResult';

filelist = dir(resultPath);
sessionPaths = regexp({filelist.name},'^#\S*.mat','match');
sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));

FI_Distance_Ratio = zeros(size(Unit,1),1);
FI_Distance_Difference = zeros(size(Unit,1),1);
FI_Distance_Relative = zeros(size(Unit,1),1);

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
        
        % FI Factor Ratio
        FI_Distance_Ratio(Unit.Session == sessionName & Unit.Cell == unit) = ...
            err_corrupted / err_original;

        % FI Factor Difference
        FI_Distance_Difference(Unit.Session == sessionName & Unit.Cell == unit) = ...
            err_corrupted - err_original;

        FI_Distance_Relative(Unit.Session == sessionName & Unit.Cell == unit) = ...
            (err_corrupted - err_original) / (err_shuffled - err_original);
    end
    fprintf('\b\b\b\b\b%02d/40', session);
end

Unit = [Unit, table(FI_Distance_Ratio, FI_Distance_Difference, FI_Distance_Relative, 'VariableNames', {'FI_Distance_Ratio', 'FI_Distance_Difference', 'FI_Distance_Relative'})];

%% Feature Importance - Event Classifier
resultPath = 'D:\Data\Lobster\BNB_Result_unitshffle.mat';

load(resultPath);

sessionNames = string(sessionNames);


FI_Event_Difference = [];
EC_Score = [];

for session = 1 : 40
    
    FI_Event_Difference = [FI_Event_Difference; result{session}.PFICrossEntropy_HWAE];
%     numBin = 40;
%     numCell = numel(result{session}.PFICrossEntropy_HEAE);
%     mean(...
%         squeeze(...
%             sum(...
%                 reshape(...
%                     permute(result{session}.feature_prob_HWAE, [3,2,1]),...
%                     numBin, numCell, 2, 5)...
%                 ,1)... % sum of all log(Prob) from all units. => joint Prob.
%             )...
%         ,3)

    EC_Score = [EC_Score;...
        repmat([...
            result{session}.balanced_accuracy_HEHW,...
            result{session}.balanced_accuracy_HEAE,...
            result{session}.balanced_accuracy_HWAE],...
            numel(result{session}.PFICrossEntropy_HWAE), 1)...
        ];
end

Unit = [Unit, table(FI_Event_Difference, EC_Score, 'VariableNames', {'FI_Event_Difference', 'EC_Score'})];
Unit.FI_Event_Difference(Unit.FI_Event_Difference < 0) = 0;

%% 

[h, p] = ttest2(...
    Unit.FI_Event_Difference((Unit.Group_HW_E ~= 1) & (Unit.EC_Score(:,6) > 0.6)),...
    Unit.FI_Event_Difference((Unit.Group_HW_E == 1) & (Unit.EC_Score(:,6) > 0.6)))



