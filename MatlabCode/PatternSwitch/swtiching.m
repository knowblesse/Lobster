%% Mode Change
truncatedTime_s = 10;
neural_data_rate = 2;
numBin = 10;

%% Load Data
SessionName = '#21JAN2-210428-195618_IL';
ML_TANK = fullfile('D:\Data\Lobster\LocationRegressionData', SessionName);
neuralData = readmatrix(fullfile(ML_TANK, glob(ML_TANK, '.*Data.csv')));
locationData = readmatrix(fullfile(ML_TANK, glob(ML_TANK, '.*buttered.csv')));
fps = readmatrix(fullfile(ML_TANK, glob(ML_TANK, 'FPS.txt')));

midPointTimes = truncatedTime_s + (1 / neural_data_rate) * (0:size(neuralData,1)-1) + 0.5 * (1 / neural_data_rate);
midPointTimes = midPointTimes';
location_rc = interp1(locationData(:,1), locationData(:,2:3), midPointTimes * fps);

BEHAV_TANK = fullfile('D:\Data\Lobster\Lobster_Recording-200319-161008\Data', SessionName);
[behaviorData, Trials, IRs, Licks, Attacks, targetdir ] = BehavDataParser(BEHAV_TANK);

load("Apparatus.mat");

numUnit = size(neuralData, 2) / numBin;

%% Label Class
% label neural data
% neuralData : real neural data
% nesting : nesting 1 

isEncounterZone = false(size(midPointTimes,1),1);
isNestingZone = false(size(midPointTimes,1),1);

for i = 1 : size(midPointTimes,1)
    isEncounterZone(i) = any((IRs(:,1) < midPointTimes(i)) & (midPointTimes(i) < IRs(:,2)));
    isNestingZone(i) = location_rc(i,2) < 200;
end

%% PCA
[coeff, score, latent] = pca(neuralData);

neural_pca = score * coeff';
neural_pca = neural_pca(:,1:2);

figure(1);
clf;
subplot(1,2,1);
hold on;
scatter(neural_pca(isEncounterZone,1), neural_pca(isEncounterZone,2), 10, 'filled', 'r');
scatter(neural_pca(~isEncounterZone,1), neural_pca(~isEncounterZone,2), 10, 'filled', 'b');

subplot(1,2,2);
hold on;
scatter(neural_pca(isNestingZone,1), neural_pca(isNestingZone,2), 10, 'filled', 'r');
scatter(neural_pca(~isNestingZone,1), neural_pca(~isNestingZone,2), 10, 'filled', 'b');

%% LDA




