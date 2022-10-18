%% LocationRegressorScripts

basePath = 'D:\Data\Lobster\LocationRegressionResult';

filelist = dir(basePath);
sessionPaths = regexp({filelist.name},'^#\S*.csv','match');
sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));

load("EmptyApparatus.mat");
wholeData = [];
% Session
for session = 1 : numel(sessionPaths)
    TANK_name = cell2mat(sessionPaths{session});
    TANK_location = char(strcat(basePath, filesep, TANK_name));
    
    % Scripts
    data = readmatrix(TANK_location);
    
    wholeData = [wholeData; data];
end

data = wholeData;
    % Calc Location Error
    locError = ((data(:,1) - data(:,7)).^2 + (data(:,2) - data(:,8)).^2).^0.5;
    
    % Cluster errors into few points
    distance = 20;
    rowEdges = 100:distance:480;
    colEdges = 20:distance:620;
    
    errorSum = zeros(numel(rowEdges), numel(colEdges));
    errorCount = zeros(numel(rowEdges), numel(colEdges));
    
    rowIndex = discretize(data(:,1), rowEdges);
    colIndex = discretize(data(:,2), colEdges);
    
    for i = 1 : size(data,1)
        try
            errorSum(rowIndex(i), colIndex(i)) = ...
                errorSum(rowIndex(i), colIndex(i)) + locError(i);
            errorCount(rowIndex(i), colIndex(i)) = ...
                errorCount(rowIndex(i), colIndex(i)) + 1;
        catch
            disp(i);
        end
    end
        
    errorCount(errorCount == 0) = 1; % TODO : no error entry gets mean error of zero
    errorMatrix = errorSum ./ errorCount;
    
    % Draw bg
    [X, Y] = meshgrid(colEdges, rowEdges);
    figure(1);
    clf;
    imshow(apparatus);
    hold on;
    colormap jet
    contour(X, Y, errorMatrix, 15);
    
    figure(1);
    clf;
    imshow(apparatus);  `
    hold on;
    colormap jet
    contour(
    
    


