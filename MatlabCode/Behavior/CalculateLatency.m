ParsedData = BehavDataParser();
latency = 0;
for i = 1 : size(ParsedData,1)
    latency = latency + ParsedData{i,3}(1);
end
latency = latency / size(ParsedData,1)