%% HECPredictiveScripts
resultPath = 'D:\Data\Lobster\EventClassificationResult_4C_Predictive_NonOverlap';

filelist = dir(resultPath);
sessionPaths = regexp({filelist.name},'^HEC_Predictive.*.mat','match');
sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));
sessionPaths = fliplr(sessionPaths);

MidPoint = zeros(11,1);
HEHW = zeros(40, 11);
HEAE = zeros(40, 11);
HWAE = zeros(40, 11);

getHEHW = @(X) X.balanced_accuracy_HEHW(2); 
getHEAE = @(X) X.balanced_accuracy_HEAE(2); 
getHWAE = @(X) X.balanced_accuracy_HWAE(2); 

for session = 1 : numel(sessionPaths)
    regResult = ...
        regexp(cell2mat(sessionPaths{session}), 'HEC_Predictive_(?<w1>.*?)_(?<w2>.*?)_NonOverlap.mat', 'names');
    load(fullfile(resultPath, cell2mat(sessionPaths{session})));
    MidPoint(session) = mean([str2double(regResult.w1), str2double(regResult.w2)]);
    HEHW(:, session) = cellfun(getHEHW, result)';
    HEAE(:, session) = cellfun(getHEAE, result)';
    HWAE(:, session) = cellfun(getHWAE, result)';
end
