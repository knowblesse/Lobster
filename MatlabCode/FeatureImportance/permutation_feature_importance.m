function pfi = permutation_feature_importance(WholeTestResult, PFITestResult)

numDatapoint = size(PFITestResult, 1); 
numUnit = size(PFITestResult, 2);
numRepeat = size(PFITestResult, 3);

pfi = zeros(numUnit,1);
for unit = 1 : numUnit
    accuracy = zeros(numRepeat, 1);
    for rep = 1 : numRepeat
        accuracy(rep) = balanced_accuracy_score(...
            WholeTestResult(:,1),...
            PFITestResult(:, unit, rep));
    end
    pfi(unit) = ...
        mean(balanced_accuracy_score(WholeTestResult(:,1), WholeTestResult(:,3)) - accuracy) / ...
        (balanced_accuracy_score(WholeTestResult(:,1), WholeTestResult(:,3)) - ... % real
        balanced_accuracy_score(WholeTestResult(:,1), WholeTestResult(:,2))); % shuffle
end
end
