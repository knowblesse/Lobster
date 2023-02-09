function pfi = permutation_feature_importance(WholeTestResult, PFITestResult, options)
arguments
    WholeTestResult;
    PFITestResult;
    options.method {mustBeMember(options.method, {'difference', 'ratio', 'relative'})} = 'difference';
end

numDatapoint = size(PFITestResult, 1); 
numUnit = size(PFITestResult, 2);
numRepeat = size(PFITestResult, 3);

original_data_accuracy = balanced_accuracy_score(WholeTestResult(:,1), WholeTestResult(:,3));
shuffled_data_accuracy = balanced_accuracy_score(WholeTestResult(:,1), WholeTestResult(:,2));

if (original_data_accuracy <= shuffled_data_accuracy)
    warning("original data accuracy is too low");
    pfi = zeros(numUnit,1);
    return 
end

pfi = zeros(numUnit,1);
for unit = 1 : numUnit
    accuracy = zeros(numRepeat, 1);
    for rep = 1 : numRepeat
        accuracy(rep) = balanced_accuracy_score(...
            WholeTestResult(:,1),...
            PFITestResult(:, unit, rep));
    end
    if strcmp(options.method, 'difference')
    pfi(unit) = ...
        mean(original_data_accuracy - accuracy);
    elseif strcmp(options.method, 'ratio')
    pfi(unit) = ...
        mean(accuracy ./ original_data_accuracy);
    elseif strcmp(options.method, 'relative')
    pfi(unit) = ...
        mean(original_data_accuracy - accuracy) / (original_data_accuracy - shuffled_data_accuracy);
    end
end
end
