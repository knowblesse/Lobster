%% AnalyzeFineDistanceData

basePath = 'D:\Data\Lobster\FineDistanceResult';
behavDataPath = 'D:\Data\Lobster\BehaviorData';
datasetDataPath = 'D:\Data\Lobster\FineDistanceDataset';

filelist = dir(basePath);
sessionPaths = regexp({filelist.name},'^#\S*.mat','match');
sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));

load("Apparatus.mat");
px2cm = 0.169;
truncatedTimes_s = 10;
neural_data_rate = 20;

%% Load Data by session
data = cell(1,40);
data_behav = cell(1,40);
fps = zeros(40,1);
midPointTimes = cell(1,40);
for session = 1 : 40
    TANK_name = cell2mat(sessionPaths{session});
    TANK_location = char(strcat(basePath, filesep, TANK_name));
    load(TANK_location); % PFITestResult, WholeTestResult(row, col, true d , shuffled d, pred d)
    load(fullfile(behavDataPath, strcat(TANK_name(1:end-19), '.mat')));
    fps(session) = readmatrix(fullfile(datasetDataPath, TANK_name(1:end-19), 'FPS.txt'));
    data{session} = WholeTestResult;
    data_behav{session} = ParsedData;
    midPointTimes{session} = truncatedTimes_s + (1/neural_data_rate)*(0:size(WholeTestResult,1)-1) + 0.5 * (1/neural_data_rate);
end

%% Compare Error btw shuffled and predicted
result1 = table(zeros(40,1), zeros(40,1), 'VariableNames',["Shuffled", "Predicted"]);
for session = 1 : 40
    result1.Shuffled(session) = mean(abs(data{session}(:,3) - data{session}(:,4))) * px2cm;
    result1.Predicted(session) = mean(abs(data{session}(:,3) - data{session}(:,5))) * px2cm;
end

%% Compare Error btw Nesting zone and Foraging zone
result2 = table(zeros(40,1), zeros(40,1), zeros(40,1), 'VariableNames', ["NestError", "ForagingError", "EncounterError"]);
for session = 1 : 40
    locError = abs(data{session}(:,3) - data{session}(:,5));
    
    isNesting = data{session}(:,2) < 200;
    
    % check isEncounter by IRsensor
    isEncounter = false(size(data{session},1),1);
    for trial = 1 : size(data_behav{session},1)
        TRON_time = data_behav{session}{trial,1}(1);
        for idxIR = 1 : size(data_behav{session}{trial,2}, 1)
            isEncounter = or(isEncounter,...
                and(...
                    midPointTimes{session} >= data_behav{session}{trial,2}(1) + TRON_time,...
                    midPointTimes{session} < data_behav{session}{trial,2}(2) + TRON_time...
                )');
        end
    end
   % check isEncounter by location
   %  isEncounter = data{session}(:,2) > 530;

    result2.NestError(session) = mean(locError(isNesting)) * px2cm;
    result2.ForagingError(session) = mean(locError(and(~isNesting, ~isEncounter))) * px2cm;
    result2.EncounterError(session) = mean(locError(isEncounter)) * px2cm;
end


%% Draw Error Heatmap
% Apparatus Image Size
accumErrorMatrix = zeros(apparatus.height, apparatus.width);
accumLocationMatrix = zeros(apparatus.height, apparatus.width);

% Run through all sessions
for session = 1 : 40
    locError = abs(data{session}(:,3) - data{session}(:,5)) * px2cm;
    for i = 1 : numel(locError)
        accumErrorMatrix(round(data{session}(i,1)), round(data{session}(i,2))) = ...
            accumErrorMatrix(round(data{session}(i,1)), round(data{session}(i,2))) + locError(i);
        
        accumLocationMatrix(round(data{session}(i,1)), round(data{session}(i,2))) = ...
            accumLocationMatrix(round(data{session}(i,1)), round(data{session}(i,2))) + 1;
    end
end

meanErrorMatrix = accumErrorMatrix ./ accumLocationMatrix;

% Location Index
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

% Method 1 : Draw Normalized Error
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

% Method 2 : Interpolate error for unknown location
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

