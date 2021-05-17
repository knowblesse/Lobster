%% BatchScript
% Script for batch running other scripts or functions

%basepath = 'D:\Lobster\GR\GR7_NeuroNexus16_v1-180607-151420\GR7';
basepath = 'D:\Lobster\20_JUN\Lobster_Recording-200319-161008\20JUN\20JUN1';
matfile_save_location = 'C:\VCF\Lobster\data\GR7\';

filelist = dir(basepath);
workingfile = regexp({filelist.name},'^#\S*','match');
workingfile = workingfile(~cellfun('isempty',workingfile));

timewindow_text = '[-8000,-6000]';
timewindow = str2num(timewindow_text);
savefile_suffix = strcat('[',num2str(timewindow(1)), ',', num2str(timewindow(2)),']');


result = zeros(numel(workingfile),2); 
for f = 1 : numel(workingfile)
    TANK_location = strcat(basepath,filesep, cell2mat(workingfile{f}));
    [ParsedData, Trials, IRs, Licks, Attacks ] = BehavDataParser(TANK_location);
    [behaviorResult, ParsedData] = analyticValueExtractor(ParsedData, false, true);
    result(f,:) = [sum(behaviorResult == 'A'), sum(behaviorResult == 'E')];
    %unitstxt2mats(strcat(TANK_location, '\recording\',ls(strcat(TANK_location, '\recording\*.txt'))));
    %AlignEvent;
    %[X, y] = generateEventClassifierDataset(TANK_location, timewindow, 100);
    %if exist(strcat(matfile_save_location,filesep,timewindow_text),'dir') == 0
    %    mkdir(strcat(matfile_save_location,filesep,timewindow_text));
    %end
    %save(strcat(matfile_save_location,filesep,timewindow_text,filesep,cell2mat(workingfile{f}),'_',savefile_suffix,'.mat'),'X','y');
end
fprintf('%s\n',timewindow_text');
fprintf('DONE\n');
