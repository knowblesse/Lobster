function balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

if numel(unique(y_true)) ~= 2
    error('Number of class is not two.');
end

classNames = unique(y_true);


balanced_accuracy = 0.5 * ...
    (sum(y_true == classNames(1) & y_pred == classNames(1)) ...
        / sum(y_true == classNames(1)) + ...
    (sum(y_true == classNames(2) & y_pred == classNames(2)) ...
        / sum(y_true == classNames(2))));
end
