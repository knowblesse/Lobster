%% Generate Time Graph

%% Constants
targetdir = 'D:\Data\Lobster\Lobster_Recording-200319-161008\20JUN1\#20JUN1-200901-105519_PL';
neural_data_rate = 2;
truncatedTime_s = 10;

%% Load Unit Data and LocationData
[~, butter_path] = glob(targetdir, 'buttered.csv');

butter = readmatrix(butter_path);

ParsedData = BehavDataParser(targetdir);

%% Load Unit Data
[Paths, ~, ~] = loadUnitData(targetdir);

%% Generate Gaussian Kernel
kernel = gausswin(ceil(KERNEL_SIZE/2)*2-1, (KERNEL_SIZE - 1) / (2 * KERNEL_STD)); % kernel size is changed into an odd number for symmetrical kernel application. see Matlab gausswin docs for the second parameter.

%% Load Unit and apply Generate Serial Data from spike timestamps(fs:1000)
numUnit = numel(Paths);
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
    % Use SessionTime_s + 10 sec data. (extra time for continuous convolution)
    % This is not mandatory, but might help to get smooth neural data. 
    serial_data = zeros( (SessionTime_s + 10) *1000,1); 
    % Before changing the serial_data, make sure that the all spikes are in SessionTime_s + 10sec
    spk = spk(spk < (SessionTime_s + 10) *1000);
    % Set value one to the spike point
    serial_data(spk,1) = 1;
    %% Convolve Gaussian kernel 
    serial_data_kerneled =  conv(serial_data,kernel,'same');
    %% Get mean and std of serialized signal from the first TRON and the last TROF
    whole_serial_data = serial_data_kerneled;
    serial_data_mean = mean(whole_serial_data);
    serial_data_std = std(whole_serial_data);
    clearvars whole_serial_data

end

TRON = ParsedData{1,1}(1);
[Paths, pathname, filename] = loadUnitData(targetdir);

TRON = ParsedData{1,1}(2);
