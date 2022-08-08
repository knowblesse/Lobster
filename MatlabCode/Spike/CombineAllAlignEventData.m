%% CombineAllAlignEventData
% Combine all aligned data

%% Define table variable for output
output = table('Size',[0, 7],'VariableTypes',["string", "string", "double", "cell", "cell", "cell", "cell"],...
    'VariableNames', ["Subject", "Session", "Cell", "Data", "RawSpikeData", "BehavData", "AE"]);
i_output = 1;

for subject = ["20JUN1", "21AUG3", "21AUG4", "21JAN2", "21JAN5"]
    basepath = strcat('D:\Data\Lobster\Lobster_Recording-200319-161008\', subject);
    filelist = dir(basepath);
    workingfile = regexp({filelist.name},'^#\S*','match');
    workingfile = workingfile(~cellfun('isempty',workingfile));
    
    for f = 1 : numel(workingfile)
        TANK_name = cell2mat(workingfile{f});
        TANK_location = char(strcat(basepath,filesep, TANK_name));
        
        % Behav Data
        ParsedData = BehavDataParser(TANK_location);
        [behaviorResult, ParsedData] = analyticValueExtractor(ParsedData, false, true);
        
        [Neurons, ~] = loadAlignedData(TANK_location);
        
        [Paths, pathname, filename] = loadUnitData(TANK_location);
        
        for n = 1 : numel(Neurons)
            load(Paths{n}); 
            if istable(SU)
                spikes = table2array(SU(:,1));
            else
                spikes = SU(:,1);
            end
            spikes = spikes * 1000;
            clearvars SU;
            
            output(i_output,:) = {subject, string(TANK_name), n, Neurons(n), spikes, ParsedData, behaviorResult};
            i_output = i_output+1;
        end
    end
end

%% Responsiveness calculation
event = 'valid_IRON';
valid_IRON_zscores = zeros(size(output,1), 40);
responsive_IRON = zeros(size(output,1),1);
for i = 1 : size(output, 1)        
    data = output.Data{i};
    valid_IRON_zscores(i, :) = data.zscore.(event);
    responsive_IRON(i) = any(abs(data.zscore.(event)) > 3);
end

fprintf('Responsive unit : %.2f %%\n', sum(responsive_IRON) / size(output,1)*100);


event = 'valid_IROF';
valid_IROF_zscores = zeros(size(output,1), 40);
responsive_IROF = zeros(size(output,1),1);
for i = 1 : size(output, 1)        
    data = output.Data{i};
    valid_IROF_zscores(i, :) = data.zscore.(event);
    responsive_IROF(i) = any(abs(data.zscore.(event)) > 3);
end

fprintf('Responsive unit : %.2f %%\n', sum(responsive_IROF) / size(output,1)*100);


% Responsieness for Avoid and Escape
event = 'valid_IROF';
responsive_A = zeros(size(output,1), 40);
responsive_E = zeros(size(output,1), 40);
for i = 1 : size(output, 1)        
    data = output.Data{i};
    responsive_A(i, :) = data.zscore.(event)(output.AE{i} == 'A',:);
    responsive_E(i, :) = data.zscore.(event)(output.AE{i} == 'E',:);
        
end

