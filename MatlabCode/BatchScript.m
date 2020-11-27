%% BatchScript
% Script for batch running other scripts or functions

basepath = 'D:\Lobster\20_JUN\Lobster_Recording-200319-161008\20JUN\20JUN1';

filelist = dir(basepath);
workingfile = regexp({filelist.name},'#\S*','match');
workingfile = workingfile(~cellfun('isempty',workingfile));

for f = 1 : numel(workingfile)
    TANK_location = strcat(basepath,filesep, cell2mat(workingfile{f}));
    %unitstxt2mats(strcat(TANK_location, '\recording\',ls(strcat(TANK_location, '\recording\*.txt'))));
    % BehavDataParser(TANK_location);
    % AlignEvent;
end
