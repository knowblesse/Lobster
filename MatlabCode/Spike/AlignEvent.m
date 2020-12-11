%% AlignEvent
% Align spike data on a specific event, compute Z-score, and save into <aligned> folder

%% PARAMETERS
TIMEWINDOW_LEFT = -1; 
TIMEWINDOW_RIGHT = +1; 
TIMEWINDOW = [TIMEWINDOW_LEFT, TIMEWINDOW_RIGHT];
TIMEWINDOW_BIN = 0.05; 
clearvars TIMEWINDOW_LEFT TIMEWINDOW_RIGHT

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

fprintf('Processing %s\n',pathname)
clearvars targetdir;

%% Find Time window in each trial
numTrial = size(ParsedData,1);
timepoint.TRON = zeros(numTrial,1);
timepoint.first_IRON = zeros(numTrial,1);
timepoint.valid_IRON = zeros(numTrial,1); % IRON leads to the first LICK
timepoint.first_LICK = zeros(numTrial,1);
timepoint.valid_IROF = zeros(numTrial,1); % IROF just before/after ATTK
timepoint.ATTK = zeros(numTrial,1);
timepoint.TROF = zeros(numTrial,1);

for t = 1 : numTrial
    start_time = ParsedData{t,1}(1);
    timepoint.TRON(t) = start_time;
    timepoint.first_IRON(t) = start_time + ParsedData{t,2}(1); 
    timepoint.valid_IRON(t) = start_time + ParsedData{t,2}(find(ParsedData{t,2}(:,1) < ParsedData{t,3}(1),1,'last'),1);
    timepoint.first_LICK(t) = start_time + ParsedData{t,3}(1);
    timepoint.valid_IROF(t) = start_time + ParsedData{t,2}(find(ParsedData{t,2}(:,1) < ParsedData{t,4}(1),1,'last'),2);
    timepoint.ATTK(t) = start_time + ParsedData{t,4}(1);
    timepoint.TROF(t) = ParsedData{t,1}(2);    
end
clearvars t start_time

%% Extract spikes
for f = 1 : numel(Paths) 
    %% Unit Data Load
    load(Paths{f}); 
    if istable(SU)
        spikes = table2array(SU(:,1));
    else
        spikes = SU(:,1);
    end
    clearvars SU;
    
    %% Spike binning
    variables = {'TRON','first_IRON','valid_IRON','first_LICK','valid_IROF','ATTK','TROF'};
    for v = variables
        eval(['Z.binned_spike.',v{1},' = zeros(numTrial,diff(TIMEWINDOW)/TIMEWINDOW_BIN);']);
        eval(['tp = timepoint.',v{1},';']);
        for t = 1 : numTrial
            spikebin = zeros(diff(TIMEWINDOW)/TIMEWINDOW_BIN,1);
            timebin = linspace(tp(t) + TIMEWINDOW(1), tp(t) + TIMEWINDOW(2),numel(spikebin) + 1);
            for tb = 1 : numel(spikebin)
                spikebin(tb) = sum(and(spikes >= timebin(tb), spikes < timebin(tb+1)));
            end
            eval(['Z.binned_spike.',v{1},'(t,:) = spikebin;']);
        end
    end
    clearvars tp tb
    
    %% Calculate Zscore
    bs = histcounts(spikes,ParsedData{1,1}(1) : TIMEWINDOW_BIN : ParsedData{end,1}(2));
    Z.mean = mean(bs);
    Z.std = std(bs);
    
    for v = variables
        eval(['Z.zscore.',v{1},' = ((sum(Z.binned_spike.',v{1},',1) ./ numTrial) - Z.mean ) ./ Z.std;']);
    end
    
    %% Session Firing Rate
    numspike = find(spikes>ParsedData{end,1}(2),1) - find(spikes>=ParsedData{1,1}(1),1);
    if isempty(numspike)
        numspike = 0;
    end
    Z.FR = numspike / ParsedData{end,1}(2) - ParsedData{1,1}(1) ; % FR btw the first TRON and the last TROF
    
    %% Trial Firing Rate
    Z.FR_trial = zeros(numTrial,1);
    for t = 1 : numTrial
        numspike = find(spikes>ParsedData{t,1}(2),1) - find(spikes>=ParsedData{t,1}(1),1);
        if isempty(numspike)
            numspike = 0;
        end
        Z.FR_trial(t) = numspike / ParsedData{t,1}(2) - ParsedData{t,1}(1); % FR btw Trial
    end
    
    %% Save
    if exist(strcat(pathname,'aligned'),'dir') == 0 % aligned 폴더가 존재하지 않으면
        mkdir(strcat(pathname,'aligned')); % 만들어줌
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

fprintf('%d 개의 파일이\n%s에 생성되었습니다.\n',f,strcat(pathname,'aligned'));
fprintf('-----------------------------------------------------------------------------\n');

clearvars f time* TIME* filename pathname Paths ParsedData
