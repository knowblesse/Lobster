%% Generate36data
% Script for making 36 attack data

basePath = 'D:\Data\Lobster\BehaviorData';

filelist = dir(basePath);
sessionPaths = regexp({filelist.name},'^#\S*','match');
sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));
fprintf('%d sessions detected.\n', numel(sessionPaths));

% Session
for session = 1 : numel(sessionPaths)
    load(fullfile(basePath, cell2mat(sessionPaths{session})));
    numTrial = size(ParsedData,1);
    isAttackIn3Sec = zeros(numTrial,1);
    for trial = 1 : numTrial
        if size(ParsedData{trial,4},1) > 1
            warning('multiple attack!');
        end
        attackTime = ParsedData{trial,4}(1) - ParsedData{trial,3}(1);

        if abs(3-attackTime) < abs(6-attackTime)
            isAttackIn3Sec(trial) = 1;
        end
    end
    fprintf('[%d] %.2f\n', session, sum(isAttackIn3Sec)/numTrial);

    save(fullfile(basePath, cell2mat(sessionPaths{session})), "isAttackIn3Sec", "Attacks", "IRs", "Licks", "ParsedData", "Trials");
end