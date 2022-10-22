%% LocationRegressorScripts

basePath = 'D:\Data\Lobster\DistanceRegressionResult';

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

    otherTank = regexp(TANK_name, '(?<f1>.*?)_distance_.*', 'names');

    xyPosition = readmatrix(fullfile('D:\Data\Lobster\LocationRegressionResult', strcat(otherTank.f1, 'result.csv')));

    data = [data; xyPosition(:,1:2), readmatrix(TANK_location)];
end

%% Draw

% Calc Location Error
locError = abs(data(:,3) - abs(data(:,5)));

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

%% Interpolate error for unknown location
x = [];
y = [];
v = [];
for row = 1 : 480
    for col = 1 : 640
        if ~isnan(meanErrorMatrix(row, col))
            x = [x, col];
            y = [y, row]; 
            v = [v, meanErrorMatrix(row, col)];
        end
    end
end

[xq, yq] = meshgrid(1:640, 1:480);
vq = griddata(x, y, v, xq, yq, 'natural');

figure(2);
clf;
mesh(xq, yq, vq);
hold on;
plot3(x, y, v, 'o');
xlim([1, 640]);
ylim([1, 480]);

figure(3);
imagesc(vq);
colormap 'jet'
colorbar
caxis([0, 400]);

figure(4);
vq(isnan(vq)) = 0;
imagesc(imgaussfilt(vq, 5, 'FilterSize', 1001));
colormap 'jet'
colorbar


%% Draw Location
accumLocationMatrix = zeros(apparatus.height, apparatus.width);
accumLocationMatrix(300, 300) = 2;
accumLocationMatrix(300, 340) = 1;


figure(1);
clf;

locationMatrix = imgaussfilt(accumLocationMatrix, 20, 'FilterSize', 1001);
locationMatrix = locationMatrix ./ max(locationMatrix, [], 'all')  .* max(accumLocationMatrix, [], 'all');
locationMatrix = locationMatrix .* apparatus.mask;

figure(4);
clf;
imshow(apparatus.image);
hold on;
colormap jet;
imagesc(locationMatrix, 'AlphaData', 0.5*ones(apparatus.height, apparatus.width));
contour(locationMatrix, 30, 'LineWidth',3);
scatter(300, 300, 'r');
scatter(340, 300, 'b');
title('Proportion of location');


% Test how location matrix is calculated
figure(1);
clf;

surf(accumLocationMatrix, 'LineStyle', 'none');
zlim([1, 100])


%% Draw Error
errorMatrix = imgaussfilt(accumErrorMatrix, 20, 'FilterSize', 101);
errorMatrix = errorMatrix .* apparatus.mask;
figure(5);
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

figure(6);
clf;
imshow(apparatus.image);
hold on;
colormap jet;
imagesc(normalizedErrorMatrix, 'AlphaData', 0.5*ones(apparatus.height, apparatus.width));
contour(normalizedErrorMatrix, 25, 'LineWidth',3);
title('Norm Error');