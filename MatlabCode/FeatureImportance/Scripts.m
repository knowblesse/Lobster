%% Script to check correlation between event classifier score of LOO and 5-fold CV
pfi_cv5 = [];

accuracy_all = zeros(40,2);
accuracy_cv5 = zeros(40,2);

for i = 1 : 40
    %pfi_cv5 = [pfi_cv5; permutation_feature_importance(result_CV5{i}.WholeTestResult_HWAE, result_CV5{i}.PFITestResult_HWAE)];
    accuracy_all(i, :) = result_All{i}.balanced_accuracy_HW(1:2);
    accuracy_cv5(i, :) = result_CV5{i}.balanced_accuracy_HWAE;
end

%% Calculate feature importance score of the Event Classifier

% HEHW = zeros(40, 2);
% HEAE = zeros(40, 2);
% HWAE = zeros(40, 2);

HEHW = [];
HEAE = [];
HWAE = [];

FI_HEHW = [];
FI_HEAE = [];
FI_HWAE = [];

for session = 1 : 40
%     HEHW(session, :) = result{session}.balanced_accuracy_HEHW;
%     HEAE(session, :) = result{session}.balanced_accuracy_HEAE;
%     HWAE(session, :) = result{session}.balanced_accuracy_HWAE;
    
    FI_HEHW = [FI_HEHW; permutation_feature_importance(result{session}.WholeTestResult_HEHW, result{session}.PFITestResult_HEHW)];
    FI_HEAE = [FI_HEAE; permutation_feature_importance(result{session}.WholeTestResult_HEAE, result{session}.PFITestResult_HEAE)];
    FI_HWAE = [FI_HWAE; permutation_feature_importance(result{session}.WholeTestResult_HWAE, result{session}.PFITestResult_HWAE)];

    numCell = size(result{session}.PFITestResult_HEHW, 2);

    HEHW = [HEHW; repmat(result{session}.balanced_accuracy_HEHW, numCell, 1)];
    HEAE = [HEAE; repmat(result{session}.balanced_accuracy_HEAE, numCell, 1)];
    HWAE = [HWAE; repmat(result{session}.balanced_accuracy_HWAE, numCell, 1)];

end

% Method 1 : SVM이 잘 먹히는 session만 따로 빼서 분석한다.
% 

crt = mean(unique(HWAE(:,1))) + std(unique(HWAE(:,1)))


[h, p] = ttest2(...
    FI_HEAE(Unit.FI_Event_Score > 0 & Unit.Session ~= "#21JAN5-210803-182450_IL" & HWAE(:,2) > crt & Unit.HEClass == 0),... 
    FI_HEAE(Unit.FI_Event_Score > 0 & Unit.Session ~= "#21JAN5-210803-182450_IL" & HWAE(:,2) > crt & Unit.HEClass == 1))