%% BatchScript
% Script for batch running other scripts or functions

basePath = 'D:\Data\Lobster\Lobster_Recording-200319-161008\Data';

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