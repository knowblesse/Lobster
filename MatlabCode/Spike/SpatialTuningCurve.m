%% SpatialTuningCurve
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
spatialMaps = zeros(632, apparatus.height, apparatus.width, 3);
spatialMaps_z = zeros(632, apparatus.height, apparatus.width);
globalCellIndex = 1;
for session = 1 : numel(sessionPaths)
    TANK_name = cell2mat(sessionPaths{session});

    % Load Neural Data
    neuralData = readmatrix(...
        fullfile(datasetBasePath, TANK_name, strcat(TANK_name, '_wholeSessionUnitData.csv')),...
        'Delimiter', ',');

    % Load Position Data
    positionData = load(glob(resultBasePath, strcat(TANK_name, '.*'), true), 'WholeTestResult');
    positionData = positionData.WholeTestResult(:, 1:2);
    
    % Number of visits per point
    numPoints = zeros(apparatus.height, apparatus.width);
    
    for datapoint = 1 : size(positionData,1)
        numPoints(round(positionData(datapoint,1)), round(positionData(datapoint,2))) = ...
            numPoints(round(positionData(datapoint,1)), round(positionData(datapoint,2))) + 1;
    end
    
    % Add all neural inputs
    numCell = size(neuralData,2);
    
    activities = zeros(numCell, apparatus.height, apparatus.width);
    
    for datapoint = 1 : size(positionData,1)
        activities(:, round(positionData(datapoint, 1)), round(positionData(datapoint, 2))) = ...
            activities(:, round(positionData(datapoint, 1)), round(positionData(datapoint, 2))) + ...
            neuralData(datapoint, :)';
    end
    
    % Add 1 to all zero elements
    numPoints(numPoints == 0) = 1;
    
    % Mean Activity
    meanActivities = activities ./ repmat(reshape(numPoints, 1, apparatus.height, apparatus.width), numCell, 1, 1);
    
    % Smoothed Activity
    smoothedActivities = zeros(numCell, apparatus.height, apparatus.width);
    for c = 1 : numCell
        smoothedActivities(c, :, :) = imgaussfilt(squeeze(meanActivities(c, :, :)), 15, 'FilterSize', 1001) .* apparatus.mask;
    end
    
    % Save maps
    for c = 1 : numCell
        img = squeeze(smoothedActivities(c, :, :));
        t = discretize(img, [-inf, linspace(-0.2, 0.2, 255), inf]);
        spatialMaps(globalCellIndex, :, :, :) = reshape(ind2rgb(t, map), 1, apparatus.height, apparatus.width, 3);
        spatialMaps_z(globalCellIndex, :, :) = smoothedActivities(c, :, :);
        globalCellIndex = globalCellIndex + 1;
    end
    
    fprintf("[%d] / %d Complete\n", session, size(sessionPaths,2));
end

%% Draw SpatialTuningCurve
for c = 101 : 200
    ax = subplot(10,10,c-100);
    map = colormap('jet');
    h = imshow(squeeze(spatialMaps(c, :, :, :)));
    set(h, 'AlphaData', apparatus.mask);
    temp = ax.Position;
    enlargeRatio = 0.1;
    ax.Position = [temp(1)-temp(3)*enlargeRatio, temp(2)-temp(4)*enlargeRatio, temp(3)*(1+enlargeRatio), temp(4)*(1+enlargeRatio)];
    caxis([-0.2,0.2]);
    title(num2str(c));
end

%% Notes on cells
% multiple peaks
cells = [51, 62, 66, 125];
% different pattern around aisle
cells = [56, 141, 79, 179];

%% Draw Example Figure
figure();
for ic = 1:4
    ax = subplot(2,2,ic);
    map = colormap('jet');
    h = imshow(squeeze(spatialMaps(cells(ic), :, :, :)));
    set(h, 'AlphaData', apparatus.mask);
    temp = ax.Position;
    enlargeRatio = 0.2;
    ax.Position = [temp(1)-temp(3)*enlargeRatio, temp(2)-temp(4)*enlargeRatio, temp(3)*(1+enlargeRatio), temp(4)*(1+enlargeRatio)];
    caxis([-0.2,0.2]);
    title(num2str(cells(ic)));
end

