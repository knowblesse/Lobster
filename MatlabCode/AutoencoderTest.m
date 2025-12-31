
neuralData = WholeTestResult(:, 3:end);
% Load Position Data
positionData = WholeTestResult(:, 1:2);

% Number of visits per point
numPoints = zeros(apparatus.height, apparatus.width);

for datapoint = 1 : size(positionData,1)
    numPoints(round(positionData(datapoint,1)), round(positionData(datapoint,2))) = ...
        numPoints(round(positionData(datapoint,1)), round(positionData(datapoint,2))) + 1;
end

% Add all neural inputs
numCell = 4;

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




figure(1);
clf;
for c = 1 : numCell
    ax = subplot(2,2,c);
    
    h = imshow(squeeze(smoothedActivities(c, :, :, :)));
    set(h, 'AlphaData', apparatus.mask);
    temp = ax.Position;
    enlargeRatio = 0.1;
    ax.Position = [temp(1)-temp(3)*enlargeRatio, temp(2)-temp(4)*enlargeRatio, temp(3)*(1+enlargeRatio), temp(4)*(1+enlargeRatio)];
    map = colormap(ax,'jet');
    caxis([0,0.5]);
    title(num2str(c));
end


