%% BatchScript
% Script for batch running other scripts or functions


basePath = 'D:\Data\Lobster\Lobster_Recording-200319-161008\';
%matfile_save_location = 'C:\Users\Knowblesse\SynologyDrive\발표\학회\2021 한국뇌신경과학회\Lobster\';

%% Subject
for subject = ["20JUN1", "21JAN2", "21JAN5", "21AUG3", "21AUG4"]
    subjectPath = strcat(basePath, subject);
    filelist = dir(subjectPath);
    sessionPaths = regexp({filelist.name},'^#\S*','match');
    sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));
    %% Session
    for session = 1 : numel(sessionPaths)
        TANK_name = cell2mat(sessionPaths{session});
        TANK_location = char(strcat(subjectPath,filesep, TANK_name));
        %% Scripts
        AlignEvent;
    end
end

fprintf('DONE\n');
