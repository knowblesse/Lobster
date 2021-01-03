%% DrawPETHforAllUnits
% Draw all units' PETH
% Created 2021JAN03 Knowblesse

%% PARAMETERS
TIMEWINDOW_LEFT = -1000; %(ms)
TIMEWINDOW_RIGHT = +1000; %(ms)
TIMEWINDOW = [TIMEWINDOW_LEFT, TIMEWINDOW_RIGHT];
clearvars TIMEWINDOW_LEFT TIMEWINDOW_RIGHT

%% Select Unit data (.mat) path
[Paths, pathname, filename] = loadUnitData();

%% Select and load EVENT data
if exist('TANK_location','var')
    [ParsedData, ~, ~, ~, ~] = BehavDataParser(TANK_location);
elseif exist(strcat(pathname,'EVENTS'),'dir') > 0 
    [ParsedData, ~, ~, ~, ~] = BehavDataParser(strcat(pathname,'EVENTS'));
else
    [ParsedData, ~, ~, ~, ~] = BehavDataParser();
end

[timepoint, numTrial] = getTimepointFromParsedData(ParsedData);

fprintf('generateEventClassifierDataset : Processing %s\n',pathname)
clearvars targetdir;

%% For every neurons, save all spikes in range with the target event
unit_data = cell(1,numel(Paths));

for f = 1 : numel(Paths) 
    unit_data{f} = cell(1,numTrial);
    
    %% Load Unit Data 
    load(Paths{f}); 
    if istable(SU)
        spikes = table2array(SU(:,1));
    else
        spikes = SU(:,1);
    end
    
    spikes = spikes * 1000; % now in ms
    clearvars SU;
    
    %% Draw Lines
    for t = 1 : numTrial
        timerange = timepoint.valid_IRON(t) + TIMEWINDOW; % Aligning Timepoint
        unit_data{f}{t} = spikes(and(timerange(1) <= spikes, spikes < timerange(2))) - timepoint.valid_IRON(t);
    end
end

%% Draw all PETH to find responsive unit
fig = figure(1);
for f = 1 : numel(Paths) 
    clf(fig);
    
    temp_axes = drawPETH(unit_data{f}, TIMEWINDOW);
    temp_axes{1}.Parent = fig;
    temp_axes{2}.Parent = fig;
    
    title(temp_axes{1},strcat('Cell : ', num2str(f)));
    drawnow;
    pause(1);
end


 
%% Redraw
fig = figure(1);
clf(fig);
f = 29;
    
temp_axes = drawPETH(unit_data{f}, TIMEWINDOW);
temp_axes{1}.Parent = fig;
temp_axes{2}.Parent = fig;

title(temp_axes{1},strcat('Cell : ', num2str(f)));
fig.Position = [-1438,478,420,370];
