%% DrawDistanceRegressionGraph

basePath = 'D:\Data\Lobster\DistanceRegressionResult';

filelist = dir(basePath);
sessionPaths = regexp({filelist.name},'^#\S*.csv','match');
sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));

load("Apparatus.mat");

%% Load Data
% Concat data from all session
% Since the distance regression result file does not contain the original
% location, get location file from LocationRegressionResult folder.

data = [];
for session = 1 : 40
    TANK_name = cell2mat(sessionPaths{session});
    TANK_location = char(strcat(basePath, filesep, TANK_name));

    otherTank = regexp(TANK_name, '(?<f1>.*?)_distance_.*', 'names');

    xyPosition = readmatrix(fullfile('D:\Data\Lobster\LocationRegressionResult', strcat(otherTank.f1, 'result.csv')));

    data = [data; xyPosition(:,1:2), readmatrix(TANK_location)];
end

%% Draw
% Calc Location Error
locError = abs(data(:,3) - data(:,5));

% Apparatus Image Size
accumErrorMatrix = zeros(apparatus.height, apparatus.width);
accumLocationMatrix = zeros(apparatus.height, apparatus.width);

for i = 1 : numel(locError)
    accumErrorMatrix(round(data(i,1)), round(data(i,2))) = ...
        accumErrorMatrix(round(data(i,1)), round(data(i,2))) + locError(i);
    
    accumLocationMatrix(round(data(i,1)), round(data(i,2))) = ...
        accumLocationMatrix(round(data(i,1)), round(data(i,2))) + 1;
end

meanErrorMatrix = accumErrorMatrix ./ accumLocationMatrix;

%% Location Index
locationMatrix = imgaussfilt(accumLocationMatrix, 20, 'FilterSize', 1001);
locationMatrix = locationMatrix .* apparatus.mask;

figure(1);
clf;
surf(accumLocationMatrix, 'LineStyle', 'none');
title('Number of visit in each pixel');

figure(2);
clf;
imshow(apparatus.image);
hold on;
colormap jet;
imagesc(locationMatrix, 'AlphaData', 0.5*ones(apparatus.height, apparatus.width));
contour(locationMatrix, 30, 'LineWidth',3);
title('Proportion of location');

%% Method 1 : Draw Normalized Error
errorMatrix = imgaussfilt(accumErrorMatrix, 20, 'FilterSize', 1001);
normalizedErrorMatrix = errorMatrix ./ locationMatrix;
normalizedErrorMatrix(isnan(normalizedErrorMatrix(:))) = 0;

normalizedErrorMatrix = normalizedErrorMatrix .* apparatus.mask;

figure(3);
clf;
imshow(apparatus.image);
hold on;
colormap jet;
imagesc(normalizedErrorMatrix, 'AlphaData', 0.5*ones(apparatus.height, apparatus.width));
contour(normalizedErrorMatrix, 25, 'LineWidth',3);
title('Norm Error');

%% Method 2 : Interpolate error for unknown location
%meanErrorMatrix = meanErrorMatrix .* apparatus.mask;
apparatus.mask(100:130, :) = 0;
x = [];
y = [];
v = [];
for row = 1 : 480
    for col = 1 : 640
        if ~isnan(meanErrorMatrix(row, col))
            if apparatus.mask(row, col) == 1
                x = [x, col];
                y = [y, row]; 
                v = [v, meanErrorMatrix(row, col)];
            end
        end
    end
end

[xq, yq] = meshgrid(1:640, 1:480);
f = scatteredInterpolant(x', y', v', 'natural', 'nearest');
vq = f(xq, yq);

figure(4);
clf;
scatter(x, y, 10, v, 'filled')
caxis([0, 400]);
colormap 'jet';
colorbar;
caxis([0, 400]);
xlim([40, 580])
ylim([120, 450]);
title('Raw error value matrix');

figure(5);
imagesc(vq);
title('Original Mean Distance L1 Error');
colormap 'jet'
colorbar
caxis([0, 400]);

figure(6);
clf;
vq(isnan(vq)) = 0;
imshow(apparatus.image);
hold on;
colormap 'jet'
imagesc(imgaussfilt(vq, 15, 'FilterSize', 1001) .* apparatus.mask, 'AlphaData', 0.3*(ones(480, 640)));
contour(imgaussfilt(vq, 15, 'FilterSize', 1001) .* apparatus.mask, 25, 'LineWidth', 3);
colorbar
title('Smoothed Mean Distance L1 Error');
