function output = loadAllUnitData()
%% LoadAllUnitData
% Load all unit data with corresponding behavior data

%% Define table variable for output
output = table('Size',[0, 8],'VariableTypes',["string", "string", "double", "cell", "cell", "cell", "cell", "cell"],...
    'VariableNames', ["Subject", "Session", "Cell", "Data", "RawSpikeData", "BehavData", "AE", "Zscore"]);
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

        % AE
        [behaviorResult, ParsedData] = analyticValueExtractor(ParsedData, false, true);
        
        %Data
        [Neurons, ~] = loadAlignedData(TANK_location);
        
        % RawSpikeData
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
            
            %Zscore
            Zscore = struct();
            Zscore.valid_IRON = Neurons{n}.zscore.valid_IRON;
            Zscore.valid_IROF = Neurons{n}.zscore.valid_IROF;
            Zscore.valid_IROF_A = ...
                ( mean(Neurons{n}.binned_spike.valid_IROF(behaviorResult == 'A',:), 1) - mean(mean(Neurons{n}.binned_spike.TRON(behaviorResult == 'A',:), 1)) )...
                ./ std(mean(Neurons{n}.binned_spike.TRON(behaviorResult == 'A',:), 1));
            Zscore.valid_IROF_E = ...
                ( mean(Neurons{n}.binned_spike.valid_IROF(behaviorResult == 'E',:), 1) - mean(mean(Neurons{n}.binned_spike.TRON(behaviorResult == 'E',:), 1)) )...
                ./ std(mean(Neurons{n}.binned_spike.TRON(behaviorResult == 'E',:), 1));

            % save
            output(i_output,:) = {subject, string(TANK_name), n, Neurons(n), spikes, ParsedData, behaviorResult, {Zscore}};
            i_output = i_output+1;
        end
    end
end
end