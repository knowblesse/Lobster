%% DR__RFI_Script

basePath = 'D:\Data\Lobster\DR_Result_RFI';

filelist = dir(basePath);
sessionPaths = regexp({filelist.name},'^#\S*.mat','match');
sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));

load("Apparatus.mat");

%% Load Data
%PFITestResult(datasize, numUnit, numRepeat)
%WholeTestResult


numRepeat = 30;
data = [];
for session = 1 : 40
    TANK_name = cell2mat(sessionPaths{session});
    TANK_location = char(strcat(basePath, filesep, TANK_name));
    load(TANK_location);
    numUnit = size(PFITestResult, 2);
    shuffled_L1_error = mean(abs(WholeTestResult(:,1) - WholeTestResult(:,2)), 1);
    predicted_L1_error = mean(abs(WholeTestResult(:,1) - WholeTestResult(:,3)), 1);
    RFI_L1_error = mean(mean(abs(repmat(WholeTestResult(:,1), 1, numUnit, numRepeat) - PFITestResult), 3), 1);
    relative_RFI_L1_error = RFI_L1_error - predicted_L1_error;
    data = [data; shuffled_L1_error, predicted_L1_error];
end