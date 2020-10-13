function unitstxt2mats(targetfile)
%% unitstxt2mats
% Converts a txt file exported from the Offline Sorter into multiple .mat files
% 2020 Knowblesse

%% Select and load txt file
if ~exist('targetfile','var')
    [FileName,PathName] = uigetfile('*.txt');
    if FileName == 0
        error('Error.unit txt file is not selected');
    end
    targetfile = strcat(PathName, FileName);
    clearvars PathName FileName
end

units = readmatrix(targetfile);

%% Segregate by Units
i_unit = 0; % index variable for accumulating unit number
for chn = 1 : 4
    num_units = max(units(units(:,2) == chn,3)); % the second column is channel, and the third column is unit number
    for un = 1 : num_units
        SU = units(and(units(:,2) == chn, units(:,3) == un),:);
        % Save file
        i_unit = i_unit + 1;
        [filepath,name,~] = fileparts(targetfile);
        save(strcat(filepath, '\', name, '_', num2str(i_unit),'.mat'),'SU');
    end
end

fprintf('----------------unitstxt2mats----------------\n');
fprintf('Output location : %s\n%d file(s) created.\n',filepath,i_unit);
end