%% LocationRegressorScripts

basePath = 'D:\Data\Lobster\LocationRegressionResult';

filelist = dir(basePath);
sessionPaths = regexp({filelist.name},'^#\S*.csv','match');
sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));

load("Apparatus.mat");

%% Load Data
% Session
data = [];
for session = 1 : 40
    TANK_name = cell2mat(sessionPaths{session});
    TANK_location = char(strcat(basePath, filesep, TANK_name));
    
    % Scripts
    data = [data; readmatrix(TANK_location)];
end

%% Draw

% Calc Location Error
locError = ((data(:,1) - data(:,5)).^2 + (data(:,2) - data(:,6)).^2).^0.5;

% Apparatus Image Size
accumErrorMatrix = zeros(apparatus.height, apparatus.width);
accumLocationMatrix = zeros(apparatus.height, apparatus.width);

for i = 1 : 1%numel(locError)
    accumErrorMatrix(round(data(i,1)), round(data(i,2))) = ...
        accumErrorMatrix(round(data(i,1)), round(data(i,2))) + locError(i);
    
    accumLocationMatrix(round(data(i,1)), round(data(i,2))) = ...
        accumLocationMatrix(round(data(i,1)), round(data(i,2))) + 1;
end

filterSigma = 20;
%% Draw Location
locationMatrix = imgaussfilt(accumLocationMatrix, filterSigma, 'FilterSize', 1001);
locationMatrix = locationMatrix .* apparatus.mask;

figure(1);
clf;
imshow(apparatus.image);
hold on;
colormap jet;
imagesc(locationMatrix, 'AlphaData', 0.5*ones(apparatus.height, apparatus.width));
contour(locationMatrix, 30, 'LineWidth',3);
title('Proportion of location');

%% Draw Error
errorMatrix = imgaussfilt(accumErrorMatrix, filterSigma, 'FilterSize', 1001);
errorMatrix = errorMatrix .* apparatus.mask;
figure(2);
clf;
imshow(apparatus.image);
hold on;
colormap jet;
imagesc(errorMatrix, 'AlphaData', 0.5*ones(apparatus.height, apparatus.width));
contour(errorMatrix, 30, 'LineWidth',3);
title('Proportion of error');

%% Draw Normalized Error
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