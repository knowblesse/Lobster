%% BatchScript
% Script for batch running other scripts or functions

%% Method 1: Subject

basePath = 'D:\Data\Lobster\Lobster_Recording-200319-161008\';

for subject = ["20JUN1", "21AUG3", "21AUG4", "21JAN2", "21JAN5"]
    subjectPath = strcat(basePath, subject);
    filelist = dir(subjectPath);
    sessionPaths = regexp({filelist.name},'^#\S*','match');
    sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));
    % Session
    for session = 1 : numel(sessionPaths)
        TANK_name = cell2mat(sessionPaths{session});
        TANK_location = char(strcat(subjectPath,filesep, TANK_name));
        % Scripts
        AlignEvent;
    end
end

fprintf('DONE\n');

%% Method 2 : All-in-one
basePath = 'F:\LobsterData';
   
filelist = dir(basePath);
sessionPaths = regexp({filelist.name},'^#\S*','match');
sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));
fprintf('%d sessions detected.\n', numel(sessionPaths));

% Session
for session = 1 : numel(sessionPaths)
    TANK_name = cell2mat(sessionPaths{session});
    TANK_location = char(strcat(basePath, filesep, TANK_name));
    % Scripts
    AlignEvent;
end
fprintf('DONE\n');