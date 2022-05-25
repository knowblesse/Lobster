function X = generateWholeSessionUnitdata(TANK_location, SessionTime_s, KERNEL_SIZE, KERNEL_STD, TIMEWINDOW_BIN, FS)
%% generateWholeSessionUnitdata
% Generate neural data snippets from the whole session
% Params
%   TANK_location : path to the tank
%   SessionTime_s : Total session time in second
%   KERNEL_SIZE : kernel size
%   KERNEL_STD : kernel std
%   TIMEWINDOW_BIN : binning window size 50ms
%   FS : number of locations per second. ex)2 = 2 locations per sec 

%% Load Unit Data
[Paths, ~, ~] = loadUnitData(TANK_location);

%% Check if the BLOF is behind the SessionTime_s
DATA = TDTbin2mat(TANK_location,'TYPE',{'epocs'});
if ~isfield(DATA.epocs,'BLOF')
    warning('generateWholeSessionUnitData : BLOF block does not exist. Using the last TROF');
    temp_time = DATA.epocs.TROF.onset(end);
else
    temp_time = DATA.epocs.BLOF.onset;
end

if temp_time < SessionTime_s
    warning('generateWholeSessionUnitData : Session time is shorter than %d sec. Using %d sec instead', SessionTime_s, round(temp_time));
    SessionTime_s = round(temp_time);
end

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

    %% Divide by 1/FS second and zscore
    idx = 1;
    for sec = 1 / FS : 1 / FS : SessionTime_s
        data = (serial_data_kerneled((sec-(1/FS))*1000+1:sec*1000) - serial_data_mean) / serial_data_std;
        % Average Binning
        % one data point during each TIMEWINDOW_BIN
        % ex) one point per 50ms
        % we acheive this by summing all 1ms point data and dividing it by TIMEWINDOW_BIN
        X(idx,(u-1)*single_unit_vector_size+1:u*single_unit_vector_size) = sum(reshape(data,TIMEWINDOW_BIN,numel(data)/TIMEWINDOW_BIN),1) / TIMEWINDOW_BIN;
        idx = idx + 1;
    end    
end
end
