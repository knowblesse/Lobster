%% DistanceTuningCurve
% Draw firing map of units

%% Dataset Paths
datasetBasePath = 'D:\Data\Lobster\FineDistanceDataset\';
resultBasePath = 'D:\Data\Lobster\FineDistanceResult_syncFixed_May';

filelist = dir(datasetBasePath);
sessionPaths = regexp({filelist.name},'^#\S*','match');
sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));
fprintf('%d sessions detected.\n', numel(sessionPaths));

addpath('..');
load('Apparatus.mat');
map = colormap('jet');

%% Batch Session

% distance min = 22.7825 23
% distance max = 615.8712 616

distanceMaps_z = zeros(632, 616);
globalCellIndex = 1;
for session = 1 : numel(sessionPaths)
    TANK_name = cell2mat(sessionPaths{session});

    % Load Neural Data
    neuralData = readmatrix(...
        fullfile(datasetBasePath, TANK_name, strcat(TANK_name, '_wholeSessionUnitData.csv')),...
        'Delimiter', ',');

    % Load Position Data
    distanceData = load(glob(resultBasePath, strcat(TANK_name, '.*'), true), 'WholeTestResult');
    distanceData = distanceData.WholeTestResult(:, 3);
    
    % Number of visits per point
    numPoints = zeros(616,1);
    
    for datapoint = 1 : size(distanceData,1)
        numPoints(round(distanceData(datapoint))) = ...
            numPoints(round(distanceData(datapoint))) + 1;
    end
    
    % Add all neural inputs
    numCell = size(neuralData,2);
    
    activities = zeros(numCell, 616);
    
    for datapoint = 1 : size(distanceData,1)
        activities(:, round(distanceData(datapoint))) = ...
            activities(:, round(distanceData(datapoint))) + neuralData(datapoint, :)';
    end
    
    % Add 1 to all zero elements
    numPoints(numPoints == 0) = 1;
    
    % Mean Activity
    meanActivities = activities ./ numPoints';
    
    % Smoothed Activity
    KERNEL_SIZE = 1000;
    KERNEL_STD = 10;
    kernel = gausswin(ceil(KERNEL_SIZE/2)*2-1, (KERNEL_SIZE - 1) / (2 * KERNEL_STD)); % kernel size is changed into an odd number for symmetrical kernel application. see Matlab gausswin docs for the second parameter.
    smoothedActivities = zeros(numCell, 616);
    for c = 1 : numCell
        smoothedActivities(c, :) = conv(meanActivities(c,:),kernel,'same');
        plot(smoothedActivities(c, :));
        ylim([-5, 5]);
        title(num2str(c));
        drawnow;
        pause(1)
    end
    
    legend;
    fprintf("[%d] / %d Complete\n", session, size(sessionPaths,2));
end




