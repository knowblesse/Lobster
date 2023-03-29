%% HECPredictiveScripts
resultPath = 'D:\Data\Lobster\EventClassificationResult_4C_Predictive_NonOverlap';

filelist = dir(resultPath);
sessionPaths = regexp({filelist.name},'^HEC_Predictive.*.mat','match');
sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));
sessionPaths = fliplr(sessionPaths);

midpoints = zeros(numel(sessionPaths),1);

HWAE_shuffle = zeros(40, numel(sessionPaths));
HWAE_original = zeros(40, numel(sessionPaths));

getHWAE_shuffle = @(X) X.balanced_accuracy_HWAE(1); 
getHWAE_original = @(X) X.balanced_accuracy_HWAE(2); 

for session = 1 : numel(sessionPaths)
    regResult = ...
        regexp(cell2mat(sessionPaths{session}), 'HEC_Predictive_(?<w1>.*?)_(?<w2>.*?)_NonOverlap.mat', 'names');
    load(fullfile(resultPath, cell2mat(sessionPaths{session})));
    midpoints(session) = mean([str2double(regResult.w1), str2double(regResult.w2)]);
    HWAE_shuffle(:, session) = cellfun(getHWAE_shuffle, result)';
    HWAE_original(:, session) = cellfun(getHWAE_original, result)';
end

%% Sort by midpoint times
[v, i] = sort(midpoints);
midpoints = midpoints(i);
HWAE_shuffle = HWAE_shuffle(:, i);
HWAE_original = HWAE_original(:,i);