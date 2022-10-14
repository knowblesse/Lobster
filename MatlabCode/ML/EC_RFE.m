%% Recursive Feature Elimination


%% Load data
load('Output_AE_RFE_max_FI.mat');
load('ClassifiedUnitData.mat');

%% Draw RFE accuracy change (one by one)
figure(1);
clf;
for session = 1 : 40
    plot(result{session}.accuracy_HW);
    ylim([0.5, 1]);
    title(num2str(session));
    drawnow;
    pause(2);
end

%% Draw RFE accuracy change
numMaxUnit = 26;
figure(1);
clf;
subplot(1,1,1);
hold on;
values = zeros(26,1);
for session = 1 : 40
    acc = result{session}.accuracy_HW;
    values = values + interp1(linspace(1, 26, numel(acc)), acc, 1:26)';
end
plot(values/40);

%% Calculate Feature Importance Measures
Units = [Units,table(zeros(size(Units,1),1), 'VariableNames',"FeatureImportance")];

sessionNames = string(sessionNames);
for session = 1 : 40
    idx = 1;
    for importantUnit = result{session}.importanceUnit_HW + 1 % change to matlab cell index
        Units.FeatureImportance(Units.Session == sessionNames{session} & Units.Cell == importantUnit) ...
            = result{session}.importanceScore_HW(idx)/result{session}.balanced_accuracy_HW(3);
        idx = idx + 1;
    end
end

%% Print Composition
for i = 1 : 3
    fprintf('Class %d : Total %d, Important %d\n',...
        i,...
        sum(Units.valid_IROF_A_type == i),...
        sum(Units.valid_IROF_A_type == i & Units.FeatureImportance > 0));
end

%% Print Accuracy
c = cell(1,4);
for i = 0 : 3
    c{i+1} = Units.FeatureImportance(Units.valid_IROF_A_type == i & Units.FeatureImportance > 0);
end
