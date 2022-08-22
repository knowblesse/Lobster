%% Constants
TIMEWINDOW = [-1000, +1000]; 
TIMEWINDOW_BIN = 50;
KERNEL_SIZE = 1000;
KERNEL_STD = 100;

basepath = 'D:\Data\Lobster\Lobster_Recording-200319-161008\Data';
outputpath = 'D:\Data\Lobster\EventClassificationData';
filelist = dir(basepath);
workingfile = regexp({filelist.name},'#\S*','match'); % read only #ed folders
workingfile = workingfile(~cellfun('isempty',workingfile));
6
for f = 1 : numel(workingfile)
    TANK_name = cell2mat(workingfile{f});
    TANK_location = fullfile(basepath, TANK_name);
    [X, y] = generateEventClassifierDataset(TANK_location, TIMEWINDOW, TIMEWINDOW_BIN, KERNEL_SIZE, KERNEL_STD);

    %% Print Output
    fprintf('[%d] Processing %s Tank\n', f, TANK_name);
    save(...
        fullfile(outputpath, strcat(TANK_name, '_eventClassificationData.mat')),...
        'X', 'y'...
    );
end
