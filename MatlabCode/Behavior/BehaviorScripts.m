%% BehaviorScripts

%% Test whether an animal shows more avoid behavior at the end of the session
% use with batch to accumulate behavior data
numTrials = zeros(40,1);
for session = 1 : 40
    numTrials(session) = numel(output{session});
end

escapeGraph = zeros(40, max(numTrials));

for session = 1 : 40
    escapeGraph(session, :) = interp1(1:numTrials(session), double(output{session}), linspace(1, numTrials(session), max(numTrials)));
end

