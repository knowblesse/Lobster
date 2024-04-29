%% review_program
% Scripts for testing feature importance

px2cm = 0.169;

%% Feature Importance - Fine Distance Regressor
resultPath = 'D:\Data\Lobster\FineDistanceResult_syncFixed_May';

filelist = dir(resultPath);
sessionPaths = regexp({filelist.name},'^#\S*.mat','match');
sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));


%%
fprintf('session : 00/40');
sessionPFI = cell(40,1);
for session = 1 : 40
    sessionName = cell2mat(regexp(cell2mat(sessionPaths{session}), '^#.*?L', 'match'));
    MAT_filename = cell2mat(sessionPaths{session});
    MAT_filePath = char(strcat(resultPath, filesep, MAT_filename));
    load(MAT_filePath); % PFITestResult, WholeTestResult(row, col, true d , shuffled d, pred d)
    
    % calculate FI for every units
    err_original = mean(abs(WholeTestResult(:,3) - WholeTestResult(:,5))) * px2cm;
    err_shuffled = mean(abs(WholeTestResult(:,3) - WholeTestResult(:,4))) * px2cm;
    
    numUnit = size(PFITestResult, 2);
    numIteration = size(PFITestResult, 3);
    
    sessionPFI_single = zeros(numIteration, numUnit);
    
    for unit = 1 : numUnit
        err_corrupted = mean(squeeze(abs(WholeTestResult(:,3) - PFITestResult(:,unit, :))), 1) * px2cm;
        sessionPFI_single(:, unit) = err_corrupted';
    end
    sessionPFI{session} = sessionPFI_single;
    fprintf('\b\b\b\b\b%02d/40', session);
end

%% Read Unit table and get FI rank
base_path = "D:\Data\Lobster\FineDistanceDataset";
sessionNames = unique(Unit.Session);
for session = 1 : 40
    [~, seq] = sort(Unit.FI_Distance(Unit.Session == sessionNames(session)));
    writematrix(seq, base_path + filesep + sessionNames(session) + filesep + 'FI_rank.csv');
end
    

