function [X, y] = generateEventClassifierDataset(tank_location, timewindow, timewindow_bin, kernel_size, kernel_std)
%% generateEventClassifierDataset()
% generate dataset for event(head entry, avoid, escape) classifier
% tank_location : string. location of tank
% timewindow : [TIMEWINDOW_LEFT(ms), TIMEWINDOW_RIGHT(ms)] default=[-1000, +1000](ms)
% timewindow_bin : bin size(ms) of the window. reshape function is used for binning. default=100(ms)
% kernel_size : size of the gaussian kernel. default= 1000(ms)
% kernel_width : width(std) of the gaussian kernel. default=100(ms)

%% Select Unit data (.mat) path
[Paths, pathname, ~] = loadUnitData(tank_location);

%% Select Tank and Load Unit and Event Data
if exist('tank_location','var')
    [ParsedData, ~, ~, ~, ~] = BehavDataParser(tank_location);
elseif exist(strcat(pathname,'EVENTS'),'dir') > 0 
    [ParsedData, ~, ~, ~, ~] = BehavDataParser(strcat(pathname,'EVENTS'));
else
    [ParsedData, ~, ~, ~, ~] = BehavDataParser();
end
[behaviorResult, ParsedData] = analyticValueExtractor(ParsedData, false, true);

fprintf('generateEventClassifierDataset : Processing %s\n',pathname)
clearvars targetdir;

%% Parse function parameters
if exist('timewindow', 'var')
    TIMEWINDOW = timewindow;
else
    TIMEWINDOW = [-1000, +1000];
end

if exist('timewindow_bin', 'var')
    TIMEWINDOW_BIN = timewindow_bin;
else
    TIMEWINDOW_BIN = 100;
end

if exist('kernel_size', 'var')
    KERNEL_SIZE = kernel_size;
else
    KERNEL_SIZE = 1000;
end

if exist('kernel_std', 'var')
    KERNEL_STD = kernel_std;
else
    KERNEL_STD = 100;
end

%% Generate Gaussian Kernel
kernel = gausswin(ceil(KERNEL_SIZE/2)*2-1, (KERNEL_SIZE - 1) / (2 * KERNEL_STD)); % kernel size is changed into an odd number for symmetrical kernel application. see Matlab gausswin docs for the second parameter.

%% Generate Array for Data
numTrial = numel(behaviorResult); 
numUnit = numel(Paths);
windowsize = diff(TIMEWINDOW);
binnedDataSize = windowsize / TIMEWINDOW_BIN;
numData = numUnit * binnedDataSize;

X = cell(numTrial*2, numUnit);
y = zeros(numTrial * 2, 1);

%% Load Unit and apply Generate Serial Data from spike timestamps(fs:1000)
for u = 1 : numUnit
    % Load Unit Data
    load(Paths{u}); 
    if istable(SU)
        spikes = table2array(SU(:,1));
    else
        spikes = SU(:,1);
    end
    clearvars SU;
    %% Serialize timestamp data(Sampling frequency = 1000Hz)
    spk = round(spikes*1000);
    % use max(10s from last spike, 10s from the last TROF) as the length of the serial data
    serial_data = zeros(max(spk(end) + (10*1000), (ceil(ParsedData{end,1}(end)) + 10)*1000),1);
    serial_data(spk,1) = 1;
    
    %% Convolve Gaussian kernel 
    serial_data_kerneled =  conv(serial_data,kernel,'same');
    
    %% Get mean and std of serialized signal and apply normalization
    serial_data_mean = mean(serial_data_kerneled);
    serial_data_std = std(serial_data_kerneled);
    whole_serial_data = (serial_data_kerneled - serial_data_mean) ./ serial_data_std;
    
    clearvars serial_data_kerneled

    %% Divide by EVENT Marker
    LICK = cell(numTrial,1);
    IROF = cell(numTrial,1);

    [timepoint, ~] = getTimepointFromParsedData(ParsedData); % onset of event (in ms)
    
    for trial = 1 : numTrial
        % Get Peri-Event Window
        LICK_window = round(TIMEWINDOW + timepoint.first_LICK(trial));
        IROF_window = round(TIMEWINDOW + timepoint.valid_IROF(trial));

        % Check if the window is out of range
        if (LICK_window(1) >= 1) && (LICK_window(2) <= numel(whole_serial_data))
            % Since the index of the whole_serial_data is actual timepoint in ms,
            % retrive the value in the window by index.
            LICK{trial} = mean(reshape(...
                whole_serial_data(LICK_window(1)+1 : LICK_window(2)),...
                TIMEWINDOW_BIN, binnedDataSize), 1);
        end

        if (IROF_window(1) >= 1) && (IROF_window(2) <= numel(whole_serial_data))
            IROF{trial} = mean(reshape(...
                whole_serial_data(IROF_window(1)+1 : IROF_window(2)),...
                TIMEWINDOW_BIN, binnedDataSize), 1);
        end
    end
    
    %% Separate Avoid and Escape
    LICK_A = LICK(behaviorResult == 'A');
    LICK_E = LICK(behaviorResult == 'E');
    IROF_A = IROF(behaviorResult == 'A');
    IROF_E = IROF(behaviorResult == 'E');
    
    %% Remove Empty Data resulted by index out of the range
    % ex. when you generate -8 ~ -6s offset data, -8 sec goes behind the exp start time in the first
    % trial. This usually does not occur in IROF dataset.
    LICK_A = LICK_A(~cellfun('isempty',LICK_A)); 
    LICK_E = LICK_E(~cellfun('isempty',LICK_E)); 
    IROF_A = IROF_A(~cellfun('isempty',IROF_A));
    IROF_E = IROF_E(~cellfun('isempty',IROF_E));
    
    %% Save Data
    % if the dataset size is reduced because of the index output the range issue, reinitialize the X
    % Main loop of this code is based on unit. So if the index issue occurs, from the first unit, the
    % size of the X will be changed. From the next unit, since the X size match,, size changing code
    % part will not run.
    if size([LICK_A; LICK_E; IROF_A; IROF_E], 1) ~= size(X,1) 
        X = cell(size([LICK_A; LICK_E; IROF_A; IROF_E], 1), numUnit);
    end
    X(:,u) = [LICK_A; LICK_E; IROF_A; IROF_E];
end

%% Generate X array
X = cell2mat(X);

%% Generate y array
y = [1*ones(numel(LICK_A),1);...
     2*ones(numel(LICK_E),1);...
     3*ones(numel(IROF_A),1);...
     4*ones(numel(IROF_E),1)];
fprintf('generateEventClassifierDataset : Complete %s\n',pathname)
end
