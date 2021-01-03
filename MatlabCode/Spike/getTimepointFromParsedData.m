function [timepoint,numTrial] = getTimepointFromParsedData(ParsedData,drawOnly)
%% Find Time window in each trial
% drawOnly : 'A' or 'E'. if provided, return only Avoid or Escape trials

if exist('drawOnly','var') > 0 
    [behaviorResult, ParsedData] = analyticValueExtractor(ParsedData, false, true);
    ParsedData = ParsedData(behaviorResult == drawOnly,:);
end

numTrial = size(ParsedData,1);
timepoint.TRON = zeros(numTrial,1);
timepoint.first_IRON = zeros(numTrial,1);
timepoint.valid_IRON = zeros(numTrial,1); % IRON leads to the first LICK
timepoint.first_LICK = zeros(numTrial,1);
timepoint.valid_IROF = zeros(numTrial,1); % IROF just before/after ATTK
timepoint.ATTK = zeros(numTrial,1);
timepoint.TROF = zeros(numTrial,1);

for t = 1 : numTrial
    start_time = ParsedData{t,1}(1) * 1000;
    timepoint.TRON(t) = start_time;
    timepoint.first_IRON(t) = start_time + ParsedData{t,2}(1) * 1000; 
    timepoint.valid_IRON(t) = start_time + ParsedData{t,2}(find(ParsedData{t,2}(:,1) < ParsedData{t,3}(1),1,'last'),1) * 1000;
    timepoint.first_LICK(t) = start_time + ParsedData{t,3}(1) * 1000;
    timepoint.valid_IROF(t) = start_time + ParsedData{t,2}(find(ParsedData{t,2}(:,1) < ParsedData{t,4}(1),1,'last'),2) * 1000;
    timepoint.ATTK(t) = start_time + ParsedData{t,4}(1) * 1000;
    timepoint.TROF(t) = ParsedData{t,1}(2) * 1000;    
end



end