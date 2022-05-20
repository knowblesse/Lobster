%% BatchScript
% Script for batch running other scripts or functions

basepath = 'D:\Data\Lobster\Lobster_Recording-200319-161008\21AUG4';
%matfile_save_location = 'C:\Users\Knowblesse\SynologyDrive\발표\학회\2021 한국뇌신경과학회\Lobster\';

filelist = dir(basepath);
workingfile = regexp({filelist.name},'^2\S*','match');
workingfile = workingfile(~cellfun('isempty',workingfile));

result = {};
for f = 1 : numel(workingfile)
    TANK_location = strcat(basepath,filesep, cell2mat(workingfile{f}));
    unitstxt2mats(strcat(TANK_location, '\recording\',ls(strcat(TANK_location, '\recording\*.txt'))));
end
fprintf('DONE\n');
