%% DrawStopInN


%% Constants
px2cm = 0.169;
BasePath_Behavior = 'D:\Data\Lobster\BehaviorData';
BasePath_FDR = 'D:\Data\Lobster\FineDistanceResult_syncFixed_May';
BasePath_Original = 'D:\Data\Lobster\Lobster_Recording-200319-161008\Data';

TIMEWINDOW_LEFT = -2000;
TIMEWINDOW_RIGHT = +2000;
numBin = 80; % 50ms bin

filelist = dir(BasePath_Original);
sessionPaths = regexp({filelist.name},'^#\S*','match');
sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));
fprintf('%d sessions detected.\n', numel(sessionPaths));

numStopped = zeros(40,1);
OUT = zeros(632, numBin);
wholeUnitIndex = 1; % general index for 632 cells
% Session
for session = 1 : numel(sessionPaths)
    TANK_name = cell2mat(sessionPaths{session});
    TANK_location = char(strcat(BasePath_Original, filesep, TANK_name));

    load(glob(BasePath_FDR, strcat(TANK_name, '.*'), true), 'WholeTestResult', 'midPointTimes');
    
    %% Get Just Stopped Index
    notMovedDistance = 5 / px2cm;% distance under this value is not considered as movement.
    timeLength_notMove = 20; % 20*50ms = 1sec % how many consecutive datapoints should meet the notMovedDistance condition
    timeLength_running = 20; 
    runningDistance = 15 / px2cm;
    justStopped = [];
    p = max(timeLength_notMove, timeLength_running*2) + 1;
    while p < size(WholeTestResult,1) - timeLength_notMove
        refPoint = WholeTestResult(p,1:2);
        testPoints = WholeTestResult(p+1:p+timeLength_notMove-1,1:2);
    
        % target condition
        % 1) after this point, the animal does not move more than
        % `notMovedDistance` for timeLength datapoints.
        % 2) animal is at least 400 pixel away from the robot (to exclude
        % E-zone stop)
        % 3) animal was at least `runningDistance` away `timeLength` datapoint
        % before (running).
        targetCondition = ...
            all(sum((testPoints - refPoint) .^2, 2).^0.5 < notMovedDistance) & ... 
            WholeTestResult(p, 3) > 400 & ...
            sum((WholeTestResult(p - timeLength_running, 1:2) - refPoint) .^2, 2).^0.5 > runningDistance/2 & ...
            sum((WholeTestResult(p - 2*timeLength_running, 1:2) - WholeTestResult(p - timeLength_running, 1:2)) .^2, 2).^0.5 > runningDistance/2;
            %sum((WholeTestResult(p - timeLength, 1:2) - refPoint) .^2, 2).^0.5 > runningDistance;
    
        if targetCondition
            justStopped = [justStopped; p];
            p = p + 2 * timeLength_notMove;
        else
            p = p + 1;
        end
    end
    numJustStopped = numel(justStopped);
    justStoppedTimes = midPointTimes(justStopped) * 1000;

    %% Load units and align to the just stopped points
    unitDataPaths = glob(fullfile(BasePath_Original, TANK_name, 'recording'), '.*mat', true);
    [Neurons, ~] = loadAlignedData(TANK_location);
    numUnit = numel(unitDataPaths);
    for unit = 1 : numUnit 
        %% Unit Data Load
        load(unitDataPaths{unit}); 
        if istable(SU)
            spikes = table2array(SU(:,1));
        else
            spikes = SU(:,1);
        end
        spikes = spikes * 1000;
        clearvars SU;
    
        %% Spike binning
        binned_spike = zeros(numJustStopped, numBin);
        for i_js = 1 : numJustStopped
            spikebin = zeros(numBin, 1);
            timebin = linspace(...
                justStoppedTimes(i_js) + TIMEWINDOW_LEFT,...
                justStoppedTimes(i_js) + TIMEWINDOW_RIGHT,...
                numBin + 1);
            for i_bin = 1 : numBin
                spikebin(i_bin) = sum(and(spikes >= timebin(i_bin), spikes < timebin(i_bin+1)));
            end
            binned_spike(i_js,:) = spikebin;
        end
        
        baseline_mean = mean(mean(Neurons{unit}.binned_spike.TRON, 1));
        baseline_std = std( mean(Neurons{unit}.binned_spike.TRON, 1) );

        mean_binned_spike = (mean(binned_spike,1) - baseline_mean) ./ baseline_std;
        
        OUT(wholeUnitIndex, :) = mean_binned_spike;
        wholeUnitIndex = wholeUnitIndex + 1;
    end

    fprintf('[%02d]\n', session);
end

%% Draw graph
% ClassifyUnits;
% load('run_and_stop.mat');
hold on;
[~, l1] = shadeplot(run_and_stop(Unit.Group_HE == 1, :), 'SD', 'sem', 'Color', [0.9922    0.3490    0.3373]);
[~, l2] = shadeplot(run_and_stop(Unit.Group_HE == 2, :), 'SD', 'sem', 'Color', [0.2588    0.7020    0.5843]);
graphs = get(gca,'Children');
legend(fliplr([graphs(1), graphs(3), graphs(6), graphs(8)]), {'HE1-HE', 'HE2-HE', 'HE1-r&s', 'HE2-r&s'})
