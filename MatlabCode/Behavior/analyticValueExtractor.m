function [behaviorResult, ParsedData] = analyticValueExtractor(ParsedData, isSeparateEscape, isKeepOnlyAE)
%% analyticValueExtractor
% Returns Analytic Value(ie. avoid, escape) of each trials based on the
% variable "ParsedData"

%% Constants
MIN_1MIN_TIMEOUT_DURATION = 55; % if an animal does not lick after this seconds from TRON, the trial is considered "Time out"
SEPARATE_3SEC_6SEC_ESCAPE = isSeparateEscape; % separate 3sec escape and 6sec escape
KEEP_ONLY_A_AND_E = isKeepOnlyAE; % keep only avoid and escape trials

%% AnalyticValues
numTrial = size(ParsedData,1);
% Avoid / Escape
behaviorResult = char(zeros(numTrial,1)); % A : Avoid | E : Escape | M : 1min timeout | G : Give up| N : No Lick
% for every trials
for trial = 1 : size(ParsedData,1)
    Trial = ParsedData{trial,1};
    IR = ParsedData{trial,2};
    Lick = ParsedData{trial,3};
    Attack = ParsedData{trial,4};
    if isempty(Attack) %  no Attack
        if diff(Trial) > MIN_1MIN_TIMEOUT_DURATION % Time out 
            behaviorResult(trial) = 'M';
        else % Give up
            behaviorResult(trial) = 'G';
        end
    else 
        nearAttackIRindex = find(IR(:,1) < Attack(1),1,'last');
        IAttackIROFI = IR(nearAttackIRindex, 2) - Attack(1);
        if IAttackIROFI >= 0 % Escape
            if SEPARATE_3SEC_6SEC_ESCAPE 
                % C : Escape on 3sec trial
                % D : Escape on 6sec trial
                if Attack(1) - Lick(1) >= 4.5 % although this value is either 3 or 6
                    behaviorResult(trial) = 'C'; % 6sec Escape
                else
                    behaviorResult(trial) = 'D'; % 3sec Escape
                end
            else
                behaviorResult(trial) = 'E';
            end
        else % Avoid
            behaviorResult(trial) = 'A';
        end
    end
end

if KEEP_ONLY_A_AND_E
    if numel(find(behaviorResult == 'G')) ~= 0
        fprintf('analyticValueExtractor : Give up Trials below. removing...\n');
        disp(find(behaviorResult == 'G'));
    end
    if numel(find(behaviorResult == 'M')) ~= 0 
        fprintf('analyticValueExtractor : Time up Trials below. removing...\n');
        disp(find(behaviorResult == 'M'));
    end
    ParsedData(behaviorResult == 'G',:) = [];
    behaviorResult(behaviorResult == 'G') = [];
    ParsedData(behaviorResult == 'M',:) = [];
    behaviorResult(behaviorResult == 'M') = [];
end
        
end
