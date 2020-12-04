%% DeleteUnitMat
% delete unit .mat files from each active(#) tanks' recording folder
% Created 2020NOV27 Knowblesse

basepath = 'D:\Lobster\20_JUN\Lobster_Recording-200319-161008\20JUN\20JUN1';

filelist = dir(basepath);
workingfile = regexp({filelist.name},'#\S*','match');
workingfile = workingfile(~cellfun('isempty',workingfile));

for f = 1 : numel(workingfile)
    TANK_location = strcat(basepath,filesep, cell2mat(workingfile{f}));
    filelist = dir(strcat(TANK_location, '\recording\*.mat'));
    for fl = 1 : size(filelist,1)
        delete(strcat(filelist(fl).folder, filesep, filelist(fl).name));
    end
    fprintf('%d files deleted in %c\n',size(filelist,1), f);
end
