%% Constants
startTime = -3000 : 200 : -1000;
endTime = startTime + 2000;

timewindows = [startTime', endTime'];
numDataset = size(timewindows, 1);

basepath = 'D:\Data\Lobster\Lobster_Recording-200319-161008\Data';
filelist = dir(basepath);
workingfile = regexp({filelist.name},'#\S*','match'); % read only #ed folders
workingfile = workingfile(~cellfun('isempty',workingfile));

outputBasepath = 'D:\Data\Lobster\EventClassificationData_4C_Predictive';
for dataset = 1 : numDataset
    TIMEWINDOW = timewindows(dataset,:);
    outputpath = fullfile(outputBasepath, strcat(...
        'HEC_Predictive_',...
        num2str(timewindows(dataset,1)),...
        '_',...
        num2str(timewindows(dataset,2))));
    mkdir(outputpath);

    TIMEWINDOW_BIN = 50;
    KERNEL_SIZE = 1000;
    KERNEL_STD = 100;
    
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
end
