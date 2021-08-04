%% BatchScript
% Script for batch running other scripts or functions

%basepath = 'D:\Data\Lobster\GR7_NeuroNexus16_v1-180607-151420\GR7';
basepath = 'D:\Data\Lobster\Lobster_Recording-200319-161008';
matfile_save_location = 'C:\Users\Knowblesse\SynologyDrive\발표\학회\2021 한국뇌신경과학회\Lobster\';

filelist = dir(basepath);
workingfile = regexp({filelist.name},'^#\S*','match');
workingfile = workingfile(~cellfun('isempty',workingfile));

timewindow_text = '[-8000,-6000]';
timewindow = str2num(timewindow_text);
savefile_suffix = strcat('[',num2str(timewindow(1)), ',', num2str(timewindow(2)),']');

result = {};
for f = 1 : numel(workingfile)
    TANK_location = strcat(basepath,filesep, cell2mat(workingfile{f}));
    datafile_loc = dir(strcat(TANK_location, filesep, 'recording', filesep, 'aligned', filesep, '*.mat'));
%    AlignEvent;
    for files = 1 : numel(datafile_loc)
        load(strcat(TANK_location, filesep, 'recording', filesep, 'aligned', filesep, datafile_loc(files).name));
        result = [result;{cell2mat(workingfile{f}),files,or(any(Z.zscore.valid_IRON >=3), any(Z.zscore.valid_IRON <=-3))}];
    end
    
%     [ParsedData, Trials, IRs, Licks, Attacks ] = BehavDataParser(TANK_location);
%     [behaviorResult, ParsedData] = analyticValueExtractor(ParsedData, false, true);
%     result(f,:) = [sum(behaviorResult == 'A'), sum(behaviorResult == 'E')];
    %unitstxt2mats(strcat(TANK_location, '\recording\',ls(strcat(TANK_location, '\recording\*.txt'))));
    %[X, y] = generateEventClassifierDataset(TANK_location, timewindow, 100);
    %if exist(strcat(matfile_save_location,filesep,timewindow_text),'dir') == 0
    %    mkdir(strcat(matfile_save_location,filesep,timewindow_text));
    %end
    %save(strcat(matfile_save_location,filesep,timewindow_text,filesep,cell2mat(workingfile{f}),'_',savefile_suffix,'.mat'),'X','y');
end
%fprintf('%s\n',timewindow_text');
fprintf('DONE\n');
