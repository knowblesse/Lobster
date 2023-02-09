function [attackSeconds] = getAttackSeconds(ParsedData)
%% getAttackSeconds
% Returns attack time for every trials. 3 or 6 
numTrial = size(ParsedData,1);
output = zeros(numTrial,1);

for trial = 1 : numTrial
    if ParsedData{trial,4}(1) > 4.5
        output(trial) = 6;
    else
        output(trial) = 3;
    end
end

