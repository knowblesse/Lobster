L%% Constants
TIMEWINDOW_BIN = 50;
KERNEL_SIZE = 1000;
KERNEL_STD = 100;
REMOVE_START_SEC = 10; % remove the top of the data
REMOVE_END_SEC = 10;
FS = 2; % location parsing sampling rate 1 : 1 loc per sec, 2 : 2 loc per sec, ...

%% Load Location Data
TANK_location = 'D:\Data\Lobster\Lobster_Recording-200319-161008\20JUN1\#20JUN1-200928-111539';
csv_path = ls(strcat(TANK_location, filesep, '*Vid1.csv'));
locationData = readAnymazeData(strcat(TANK_location, filesep, csv_path));

%% Convert Location Data to 1Hz
data_length = floor(locationData(end,1) * FS);
y = zeros(data_length, 2);

idx = 1;
for sec = 1 / FS : 1 / FS : floor(locationData(end,1))
    start_idx = find(locationData(:,1)>=sec-(1/FS),1,'first');
    end_idx = find(locationData(:,1)<sec,1,'last');
    y(idx,:) = mean(locationData(start_idx:end_idx,2:3),1);
    idx = idx + 1;
end

%% Interpolate NaN
% Whenever the tracker lost the animal, it does not record the location of the animal. 
% This result NaN when you do the mean(locationData). 
if sum(sum(isnan(y))) ~= 0 
    fprintf('%d NaN detected. Using linear interpolation to compensate\n', sum(sum(isnan(y))));
    y1 = y(:,1);
    y2 = y(:,2);
    nan_index_y1 = isnan(y1);
    nan_index_y2 = isnan(y2);
    r = 1 : size(y,1);
    y1(nan_index_y1) = interp1(r(~nan_index_y1), y1(~nan_index_y1), r(nan_index_y1));
    y2(nan_index_y2) = interp1(r(~nan_index_y2), y2(~nan_index_y2), r(nan_index_y2));
    y = [y1,y2];
end

%% Load Unit Data
[Paths, pathname, filename] = loadUnitData(TANK_location);

%% Generate Gaussian Kernel
kernel = gausswin(ceil(KERNEL_SIZE/2)*2-1, (KERNEL_SIZE - 1) / (2 * KERNEL_STD)); % kernel size is changed into an odd number for symmetrical kernel application. see Matlab gausswin docs for the second parameter.

%% Load Unit and apply Generate Serial Data from spike timestamps(fs:1000)
numUnit = numel(Paths);
single_unit_vector_size_ms = (1/FS)*1000 / TIMEWINDOW_BIN;
X = zeros(data_length,single_unit_vector_size_ms*numUnit); 
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
    serial_data = zeros(data_length * 1000 + (10*1000),1); 
    serial_data(spk,1) = 1;
    %% Convolve Gaussian kernel 
    conv_ =  conv(serial_data,kernel);
    % Trim start/end point of the data to match the size
    serial_data_kerneled = conv_(...
         1  + (ceil(KERNEL_SIZE/2)*2-2)/2 :...
        end - (ceil(KERNEL_SIZE/2)*2-2)/2);
    %% Get mean and std of serialized signal from the first TRON and the last TROF
    whole_serial_data = serial_data_kerneled;
    serial_data_mean = mean(whole_serial_data);
    serial_data_std = std(whole_serial_data);
    clearvars whole_serial_data

    %% Divide by 1/FS second and zscore
    idx = 1;
    for sec = 1 / FS : 1 / FS : data_length / FS
        data = (serial_data_kerneled((sec-(1/FS))*1000+1:sec*1000) - serial_data_mean) / serial_data_std;
        % Average Binning
        X(idx,(u-1)*single_unit_vector_size_ms+1:u*single_unit_vector_size_ms) = sum(reshape(data,TIMEWINDOW_BIN,numel(data)/TIMEWINDOW_BIN),1) / TIMEWINDOW_BIN;
        idx = idx + 1;
    end    
end

%% Print Output
Tank_name = cell2mat(regexp(TANK_location,'.+\\(?:.+\\)*(.+$)','tokens', 'once'));
X_data_path = strcat(TANK_location, filesep, regexp(Tank_name,'[^#].*','match','once'), '_regressionData_X.csv');
y_data_path = strcat(TANK_location, filesep, regexp(Tank_name,'[^#].*','match','once'), '_regressionData_y.csv');

if (sum(sum(isnan(X(REMOVE_START_SEC * FS : end - REMOVE_END_SEC * FS,:)))) ~= 0) || (sum(sum(isnan(y(REMOVE_START_SEC * FS : end - REMOVE_END_SEC * FS,:)))) ~= 0)
    error('nan is in the data!');
end

writematrix(X(REMOVE_START_SEC * FS: end - REMOVE_END_SEC * FS,:),X_data_path,'Delimiter',',');
writematrix(y(REMOVE_START_SEC * FS: end - REMOVE_END_SEC * FS,:),y_data_path,'Delimiter',',');

fprintf(repmat('-',1,64));
fprintf('\nLocation Regression Dataset Generator\n');
fprintf(strcat(Tank_name, " : Processed with ", num2str(numUnit), ' cells\n'));
fprintf("File saved : %s\n", X_data_path);
fprintf("File saved : %s\n", y_data_path);
