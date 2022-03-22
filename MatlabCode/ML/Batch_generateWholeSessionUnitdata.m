%% Constants
TIMEWINDOW_BIN = 50;
SessionTime_s = 30*60;
KERNEL_SIZE = 1000;
KERNEL_STD = 100;
REMOVE_START_SEC = 10; % remove the top of the data
REMOVE_END_SEC = 10;
FS = 2; % location parsing sampling rate 1 : 1 loc per sec, 2 : 2 loc per sec, ...

basepath = 'D:\Data\Lobster\Lobster_Recording-200319-161008\20JUN1';
filelist = dir(basepath);
workingfile = regexp({filelist.name},'^#\S*','match');
workingfile = workingfile(~cellfun('isempty',workingfile));

for f = 1 : numel(workingfile)
    
    TANK_location = strcat(basepath, filesep, workingfile{f}{1});
    X = generateWholeSessionUnitdata(TANK_location, SessionTime_s, KERNEL_SIZE, KERNEL_STD, TIMEWINDOW_BIN, FS);

    %% Print Output
    Tank_name = cell2mat(regexp(TANK_location,'.+\\(?:.+\\)*(.+$)','tokens', 'once'));
    X_data_path = strcat(TANK_location, filesep, regexp(Tank_name,'[^#].*','match','once'), '_wholeSessionUnitData.csv');

    if (sum(sum(isnan(X(REMOVE_START_SEC * FS : end - REMOVE_END_SEC * FS,:)))) ~= 0)
        error('nan is in the data!');
    end

    writematrix(X(REMOVE_START_SEC * FS+1: end - REMOVE_END_SEC * FS,:),X_data_path,'Delimiter',',');
end