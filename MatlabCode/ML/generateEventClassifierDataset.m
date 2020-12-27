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
    conv_ =  conv(serial_data,kernel);
    % Trim start/end point of the data to match the size
    serial_data_kerneled = conv_(...
         1  + (ceil(KERNEL_SIZE/2)*2-2)/2 :...
        end - (ceil(KERNEL_SIZE/2)*2-2)/2);
    %% Get mean and std of serialized signal from the first TRON and the last TROF
    whole_serial_data = serial_data_kerneled(round(ParsedData{1,1}(1)*1000) : round(ParsedData{end,1}(end)*1000));
    serial_data_mean = mean(whole_serial_data);
    serial_data_std = std(whole_serial_data);
    clearvars whole_serial_data

    %% Divide by EVENT Marker
    IRON = cell(numTrial,1);
    IROF = cell(numTrial,1);
    for t = 1 : numTrial
        % Get event time
        tron_time = ParsedData{t,1}(1) * 1000;
        iron_time = ParsedData{t,2}(1) * 1000 + tron_time;%first iron
        nearAttackIRindex = find(ParsedData{t,2}(:,1) < ParsedData{t,4}(1), 1, 'last');
        irof_time = ParsedData{t,2}(nearAttackIRindex, 2) * 1000 + tron_time;
        % Get range of analysis
        iron_time_range = TIMEWINDOW + iron_time;
        irof_time_range = TIMEWINDOW + irof_time;
        % Splice by the range. exclude the last element to match the size. 
        % and apply zscore
        iron_data = (serial_data_kerneled(round(iron_time_range(1)) : round(iron_time_range(2))-1) - serial_data_mean) / serial_data_std;
        irof_data = (serial_data_kerneled(round(irof_time_range(1)) : round(irof_time_range(2))-1) - serial_data_mean) / serial_data_std;
        % Average Binning
        IRON{t} = sum(reshape(iron_data,TIMEWINDOW_BIN,binnedDataSize),1) / TIMEWINDOW_BIN;
        IROF{t} = sum(reshape(irof_data,TIMEWINDOW_BIN,binnedDataSize),1) / TIMEWINDOW_BIN;
    end
    clearvars tron_time iron_* irof_*
    
    %% Separate Avoid and Escape
    IROF_A = IROF(behaviorResult == 'A');
    IROF_E = IROF(behaviorResult == 'E');
    clearvars IROF
    
    %% Save Data
    X(:,u) = [IRON;IROF_A;IROF_E];
end

%% Generate X array
X = cell2mat(X);

%% Generate y array
y = [1*ones(numel(IRON),1);...
     2*ones(numel(IROF_A),1);...
     3*ones(numel(IROF_E),1)];
fprintf('generateEventClassifierDataset : Complete %s\n',pathname)
end
