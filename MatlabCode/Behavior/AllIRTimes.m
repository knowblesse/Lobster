%% BatchScript
% Script for batch running other scripts or functions

basePath = 'D:\Data\Lobster\BehaviorData';

filelist = dir(basePath);
sessionPaths = regexp({filelist.name},'^#\S*','match');
sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));
fprintf('%d sessions detected.\n', numel(sessionPaths));


allHWTimes_6 = [];
allHWTimes_3 = [];
% Session
for session = 1 : 40
    TANK_name = cell2mat(sessionPaths{session});
    TANK_location = char(strcat(basePath, filesep, TANK_name));
    % Scripts
    load(fullfile(basePath, TANK_name));
    
    numTrial = size(ParsedData,1);
    for trial = 1 : numTrial
        % Find valid IROF
        nearAttackIRindex = find(ParsedData{trial,2}(:,1) < ParsedData{trial,4}(1), 1, 'last');
        HWTime = ParsedData{trial,2}(nearAttackIRindex,2) - ParsedData{trial,3}(1);
        if isAttackIn3Sec(trial) == 1
            allHWTimes_3 = [allHWTimes_3; HWTime];
        else
            allHWTimes_6 = [allHWTimes_6; HWTime];
        end
    end
end
fprintf('DONE\n');