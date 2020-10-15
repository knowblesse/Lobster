%% TreeFertilizer
% Generate X,y dataset for selected sessionfolder.
%% Constants
TIMEWINDOW_LEFT = -2; % 이벤트를 기점으로 몇초 전 데이터까지 사용할지.(s)
TIMEWINDOW_RIGHT = +2; % 이벤트를 기점으로 몇포 후 데이터를 사용할지.(s)
TIMEWINDOW_BIN = 100; % TIMEWINDOW의 각각의 bin 크기는 얼마로 잡을지.(ms)
stdev = 150; % std of Gausswin (ms)
kernel_size = 1500;% (ms)
windowlength = (TIMEWINDOW_RIGHT - TIMEWINDOW_LEFT) * 1000;
binnedDataSize = windowlength/TIMEWINDOW_BIN; % divied by 100ms

%% Select Session Folder (ex. 0704)
dir_sess = uigetdir('','Select Session folder');

%% Find Event Folder
if exist(strcat(dir_sess,'\EVENTS'),'dir') == 7 
    dir_even = strcat(dir_sess,'\EVENTS');
else
    error('EVENT folder is not in the sessiondir location');
end

%% Find Unit Data
% loc0 = dir(strcat(dir_sess,'\*.mat'));
% name_neur = {loc0.name}; % neuron names
% for l = 1 : numel(name_neur)
%     if contains(name_neur(l), 'UNIT') % delete *UNIT.mat file
%         name_neur(l) = [];
%     end
% end
%clear loc0 l
name_neur = uigetfile('*.mat','Select Unit.mat Files','MultiSelect','on');
fprintf('%d Units are found\n',numel(name_neur));

%% Parse Event
[ParsedData, ~, ~, ~, ~ ] = BehavDataParser(dir_even);
numTrial = size(ParsedData,1);
IR_window = zeros(0,2); % [first IRON, Last IROF]
excluded_trial_index = [];
for t = 1 : numTrial
    if isempty(ParsedData{t,3}) % No Lick
        excluded_trial_index = [excluded_trial_index, t];
        continue % skip
    end
    absoluteTime = ParsedData{t,1}(1);
    tempIR_window = [ParsedData{t,2}(1), ParsedData{t,2}(end)] + absoluteTime;
    IR_window = [IR_window;tempIR_window];
end
clearvars t absoluteTime tempIR_window 

%% Exclude no lick trial info
numTrial = size(IR_window,1);
ParsedData(excluded_trial_index,:) = []; 
clearvars excluded_trial_index

%% Label Avoid and Escape Trial
behaviorResult = AnalyticValueExtractor(ParsedData, false, true);
    % isKeepOnlyAE = true ==> possibly no effect, because no Lick trials
    % are removed.
if numel(behaviorResult) ~= numTrial
    error('Trial number mismatch');
end

%% Iterate each unit data
X = cell(numTrial*2, numel(name_neur));
for f = 1 : numel(name_neur) % for each selected units, 
    %% Load and save spike timepoints
    load(strcat(dir_sess,'\',name_neur{f}));
    spikes = table2array(SU(:,1));
    clearvars SU;
    
    %% Calculate Intertrial firing rate
    total_time = 0;
    total_spikes = 0;
    for t = 1 : numTrial
        total_time = total_time + ...
            (ParsedData{t,1}(end) - ParsedData{t,1}(1));
        total_spikes = total_spikes + ...
            sum(and(spikes >= ParsedData{t,1}(1),...
                    spikes <  ParsedData{t,1}(end)...
                    )...
                );
    end
    FR = total_spikes / total_time;
    
    clearvars total_spikes total_time t
    
    %% Make psudo serial data(fs:1000)
    spk = round(spikes*1000);
    % length of the pseudo_serial_data = 
    % max(add 10sec to last spike, add 10sec to end of the final trial)
    length_pseudo_serial_data = max(spk(end)+10000, ParsedData{end,1}(end)*1000+10000);
    pseudo_serial_data = zeros(length_pseudo_serial_data,1); 
    pseudo_serial_data(spk,1) = 1;
    clearvars length_pseudo_serial_data spk
    
    % Generate gaussian kernel
    kernel = gausswin(kernel_size,kernel_size/(2*stdev));
    % Convolve gaussian kernel
    tempdata = conv(pseudo_serial_data,kernel);
    % Trim start/end point of the data to match the size
    error('TODO : spike_kerneled and pseudo_serial_data mismatch!');
    spike_kerneled = tempdata(kernel_size/2+1:end-kernel_size/2+1);
    
    %% Divide by EVENT Marker
    IRON = cell(numTrial,1);
    IROF = cell(numTrial,1);
    for t = 1 : numTrial
        % Get event time
        tron_time = ParsedData{t,1}(1);
        iron_time = ParsedData{t,2}(1) + tron_time;
        irof_time = ParsedData{t,2}(end) + tron_time;
        % Get range of analysis
        iron_time_range = [TIMEWINDOW_LEFT, TIMEWINDOW_RIGHT] + iron_time;
        irof_time_range = [TIMEWINDOW_LEFT, TIMEWINDOW_RIGHT] + irof_time;
        % Splice by the range
        iron_data = spike_kerneled(round(iron_time_range(1)*1000) : round(iron_time_range(2)*1000));
        irof_data = spike_kerneled(round(irof_time_range(1)*1000) : round(irof_time_range(2)*1000));
        iron_data = iron_data(1:windowlength);
        irof_data = irof_data(1:windowlength);
        % Binning
        IRON{t} = sum(reshape(iron_data,TIMEWINDOW_BIN,binnedDataSize),1);
        IROF{t} = sum(reshape(irof_data,TIMEWINDOW_BIN,binnedDataSize),1);
    end
    clearvars tron_time iron_* irof_*
    clearvars kernel tempdata pseudo_serial_data spikes
    
    %% Get mean and std for all binned data
    error('TODO : why multiple -1000');
    all_data = -1000*ones(numTrial*2*binnedDataSize,1);
    for t = 1 : numTrial
        all_data(2*binnedDataSize*(t-1)+1:2*binnedDataSize*t) = ...
            [IRON{t}, IROF{t}];
    end
    m = mean(all_data);
    s = std(all_data);
    clearvars all_data
    
    %% Apply Z transform
    for t = 1 : numTrial
        IRON{t} = (IRON{t} - m) / s;
        IROF{t} = (IROF{t} - m) / s;
    end
    clearvars m s
    
    %% Separate Avoid and Escape
    IROF_A = IROF(behaviorResult == 'A');
    IROF_E = IROF(behaviorResult == 'E');
    clearvars IROF
    
    %% Save Data
    X(:,f) = [IRON;IROF_A;IROF_E];
end

%% Generate X array
X = cell2mat(X);

%% Generate y array
y = [1*ones(numel(IRON),1);...
     2*ones(numel(IROF_A),1);...
     3*ones(numel(IROF_E),1)];

clearvars -except X y