%% AlignEvent
% Align spike data on a specific event, compute Z-score, and save into <aligned> folder

%% PARAMETERS
TIMEWINDOW_LEFT = -1000; %(ms)
TIMEWINDOW_RIGHT = +1000; %(ms)
TIMEWINDOW_BIN = 50; %(ms) 
numBin = (TIMEWINDOW_RIGHT - TIMEWINDOW_LEFT)/TIMEWINDOW_BIN; % number of bins

%% Select Unit data (.mat) path
[Paths, pathname, filename] = loadUnitData(TANK_location);

% Sucrose Sessions (nearly deprecated)
if contains(pathname,'suc') % if path name has 'suc' in it, consider it as sucrose training (no attk) data
    isSuc = true;
else
    isSuc = false;
end

%% Select and load EVENT data
if exist('TANK_location','var')
    [ParsedData, ~, ~, ~, ~] = BehavDataParser(TANK_location);
elseif exist(strcat(pathname,'EVENTS'),'dir') > 0 
    [ParsedData, ~, ~, ~, ~] = BehavDataParser(strcat(pathname,'EVENTS'));
else
    [ParsedData, ~, ~, ~, ~] = BehavDataParser();
end

[timepoint, numTrial] = getTimepointFromParsedData(ParsedData);

fprintf('AlignEvent : Processing %s\n',pathname)
clearvars targetdir;

%% Extract spikes
for f = 1 : numel(Paths) 
    %% Unit Data Load
    load(Paths{f}); 
    if istable(SU)
        spikes = table2array(SU(:,1));
    else
        spikes = SU(:,1);
    end
    spikes = spikes * 1000;
    clearvars SU;
    
    %% Spike binning
    event_list = ["TRON","first_IRON","valid_IRON","first_LICK","valid_IROF","ATTK","TROF"];
    for event = event_list
        Z.binned_spike.(event) = zeros(numTrial, numBin);
        for trial = 1 : numTrial
            spikebin = zeros(numBin, 1);
            timebin = linspace(...
                timepoint.(event)(trial) + TIMEWINDOW_LEFT,...
                timepoint.(event)(trial) + TIMEWINDOW_RIGHT,...
                numBin + 1);
            for i_bin = 1 : numBin
                spikebin(i_bin) = sum(and(spikes >= timebin(i_bin), spikes < timebin(i_bin+1)));
            end
            Z.binned_spike.(event)(trial,:) = spikebin;
        end
    end
        
    clearvars event trial i_bin
    
    %% Calculate Zscore
    % For the Z score calculation, meaning bins across whole trial must come first
    % before applying the z transformation. (mean of z scores are not z score)
    % I used the +- 1 seconds around TRON as the baseline. 
    % To make the mean and the std as representative as possible, I did not use
    % one single timepoint as a baseline, rather used "every +- 1 seconds around TRON".
    % After meaning the binned spikes from the baseline period across trials, mean and std of bin 
    % is calculated. And then, this value is used for z transformation to "binned_spike"s aligned to
    % all behavior event. We might lose all TRON responsive neurons, but since TRON is not my 
    % Event Of Interest, it would be okay.
    
    Z.mean_baseline = mean(mean(Z.binned_spike.TRON,1));
    Z.std_baseline = std(mean(Z.binned_spike.TRON,1));
    
    %% Z score calculation
    for event = event_list
        Z.zscore.(event) = ( mean(Z.binned_spike.(event),1) - Z.mean_baseline) ./ Z.std_baseline;
    end
    
    %% Session Firing Rate
    numspike = find(spikes>ParsedData{end,1}(2) * 1000,1) - find(spikes>=ParsedData{1,1}(1) * 1000,1);
    if isempty(numspike)
        numspike = 0;
    end
    Z.FR = numspike / (ParsedData{end,1}(2) - ParsedData{1,1}(1)) ; % FR btw the first TRON and the last TROF
    
    %% Trial Firing Rate
    Z.FR_trial = zeros(numTrial,1);
    for t = 1 : numTrial
        numspike = find(spikes>ParsedData{t,1}(2) * 1000,1) - find(spikes>=ParsedData{t,1}(1) * 1000,1);
        if isempty(numspike)
            numspike = 0;
        end
        Z.FR_trial(t) = numspike / (ParsedData{t,1}(2) - ParsedData{t,1}(1)); % FR btw Trial
    end
    
    %% Save
    if exist(strcat(pathname,'aligned'),'dir') == 0 % if aligned folder does not exist,
        mkdir(strcat(pathname,'aligned')); % make one
    end
    % parse filename
    filename_date = regexp(filename{f}, '\d{6}-\d{6}_eTe1*','match');
    filename_date = filename_date{1}(3:6);
    filename_cellnum = regexp(filename{f}, '_\d{1,}.mat','match');
    filename_cellnum = filename_cellnum{1}(2:end - numel('.mat'));
    if isempty(filename_date) || isempty(filename_cellnum)
        error("File name parsing failed");
    end
    %% Save Data
    % save data : original data location
    save([pathname,'\aligned\',filename_date,'_',filename_cellnum,'_aligned.mat'],'Z');
    clearvars filename_date temp1 temp2 filename_cellnum Z 
end

fprintf('%d files are created at \n%s\n',f,strcat(pathname,'aligned'));
fprintf('-----------------------------------------------------------------------------\n');

clearvars f time* TIME* filename pathname Paths ParsedData
