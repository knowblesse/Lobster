load("ClassifiedUnitData.mat");

%% Draw classification confusion matrix
load("Output_AE.mat")

sessionNames = string(sessionNames);
PLdata = result(contains(sessionNames, "PL"));
ILdata = result(contains(sessionNames, "IL"));


ba_HE = zeros(40, 2);
ba_HW = zeros(40, 2);

for session = 1 : 40
    ba_HE(session,:) = result{session}.balanced_accuracy_HE;
    ba_HW(session,:) = result{session}.balanced_accuracy_HW;
end

%% PL
ba_HE_PL = zeros(numel(PLdata), 2);
ba_HW_PL = zeros(numel(PLdata), 2);

for session = 1 : numel(PLdata)
    ba_HE_PL(session, :) = PLdata{session}.balanced_accuracy_HE;
    ba_HW_PL(session, :) = PLdata{session}.balanced_accuracy_HW;
end

%% IL
ba_HE_IL = zeros(numel(ILdata), 2);
ba_HW_IL = zeros(numel(ILdata), 2);

for session = 1 : numel(ILdata)
    ba_HE_IL(session, :) = ILdata{session}.balanced_accuracy_HE;
    ba_HW_IL(session, :) = ILdata{session}.balanced_accuracy_HW;
end


%% HE HW
load("Output_HEHW.mat");
ba = zeros(40, 2);
for session = 1 : 40
    ba(session,:) = result{session}.balanced_accuracy;
end