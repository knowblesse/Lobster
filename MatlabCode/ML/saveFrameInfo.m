%% saveFrameInfo
% Batch script to read the frame info from TDT binary and save it into mat file

basePath = 'D:\Data\Lobster\Lobster_Recording-200319-161008\Data';
outputBasePath = 'D:\Data\Lobster\FineDistanceDataset';

filelist = dir(basePath);
sessionPaths = regexp({filelist.name},'^#\S*','match');
sessionPaths = sessionPaths(~cellfun('isempty',sessionPaths));
fprintf('%d sessions detected.\n', numel(sessionPaths));

% Session
for session = 1 : numel(sessionPaths)
    TANK_name = cell2mat(sessionPaths{session});
    TANK_location = char(strcat(basePath, filesep, TANK_name));
    % Scripts
    DATA = TDTbin2mat(TANK_location, 'TYPE', {'epocs'});
    Vid1 = DATA.epocs.Vid1;

    frameNumber = Vid1.data;
    frameTime = Vid1.onset;

    save_location = fullfile(outputBasePath, TANK_name, strcat(TANK_name, '_frameInfo.mat'));
    save(save_location, 'frameNumber', 'frameTime');
end
fprintf('DONE\n');