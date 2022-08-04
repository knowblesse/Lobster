%% Constants
TIMEWINDOW_BIN = 50;
SessionTime_s = 30*60;
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

%% Make X data
X = generateWholeSessionUnitdata(TANK_location, SessionTime_s, KERNEL_SIZE, KERNEL_STD, TIMEWINDOW_BIN, FS);
% TODO : the size of this data does not match with the y data. 

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
