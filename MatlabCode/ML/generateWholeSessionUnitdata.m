function X = generateWholeSessionUnitdata(TANK_location, SessionTime_s, KERNEL_SIZE, KERNEL_STD, TIMEWINDOW_BIN, FS)

%% Load Unit Data
[Paths, ~, ~] = loadUnitData(TANK_location);

%% Generate Gaussian Kernel
kernel = gausswin(ceil(KERNEL_SIZE/2)*2-1, (KERNEL_SIZE - 1) / (2 * KERNEL_STD)); % kernel size is changed into an odd number for symmetrical kernel application. see Matlab gausswin docs for the second parameter.

%% Load Unit and apply Generate Serial Data from spike timestamps(fs:1000)
numUnit = numel(Paths);
single_unit_vector_size = (1/FS)*1000 / TIMEWINDOW_BIN;
X = zeros(SessionTime_s*FS,single_unit_vector_size*numUnit); 
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
    % use 10 sec from the last video time as the length of the serial data
    serial_data = zeros(SessionTime_s * 1000 + (10*1000),1); 
    serial_data(spk,1) = 1;
    %% Convolve Gaussian kernel 
    serial_data_kerneled =  conv(serial_data,kernel,'same');
    %% Get mean and std of serialized signal from the first TRON and the last TROF
    whole_serial_data = serial_data_kerneled;
    serial_data_mean = mean(whole_serial_data);
    serial_data_std = std(whole_serial_data);
    clearvars whole_serial_data

    %% Divide by 1/FS second and zscore
    idx = 1;
    for sec = 1 / FS : 1 / FS : SessionTime_s
        data = (serial_data_kerneled((sec-(1/FS))*1000+1:sec*1000) - serial_data_mean) / serial_data_std;
        % Average Binning
        % mean Z value during TIMEWINDOW ms ---> one data point
        % single_unit_vector_size data poins per unit
        X(idx,(u-1)*single_unit_vector_size+1:u*single_unit_vector_size) = sum(reshape(data,TIMEWINDOW_BIN,numel(data)/TIMEWINDOW_BIN),1) / TIMEWINDOW_BIN;
        idx = idx + 1;
    end    
end
end