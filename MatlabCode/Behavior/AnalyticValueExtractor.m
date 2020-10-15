function behaviorResult = analyticValueExtractor(ParsedData, isSeparateEscape, isKeepOnlyAE)
%% AnalyticValueExtractor
% Returns Analytic Value(ie. avoid, escape) of each trials based on the
% variable "ParsedData"

%% Constants
% MIN_LICK_CLUSTER_INTERVAL = 0.3; % 히스토그램 분석후 97.6035% 가 속하는 0.3초로 정함.
% MIN_IR_CLUSTER_INTERVAL = 1.3; % 히스토그램 분석후 67.8255% 가 속하는 1.3초로 정함.
MIN_1MIN_TIMEOUT_DURATION = 55; % 1min timeout으로 인정하기 위한 최소의 시간.
SEPARATE_3SEC_6SEC_ESCAPE = isSeparateEscape; % 3초 escape과 6초 escape을 구별할지.
KEEP_ONLY_A_AND_E = isKeepOnlyAE; % Avoid와 Escape이 아닌 trial은 모두 제거.

%% AnalyticValues
numTrial = size(ParsedData,1);
% Clusters
% numLickClusters = zeros(numTrial,1);
% numIRClusters = zeros(numTrial,1);
% Avoid / Escape
behaviorResult = char(zeros(numTrial,1)); % A : Avoid | E : Escape | M : 1min timeout | G : Give up| N : No Lick
% for every trials
for trial = 1 : size(ParsedData,1)
    IRs = ParsedData{trial,2};
    Licks = ParsedData{trial,3};
%     %% IR Cluster
%     if isempty(IRs) % IR 이 하나도 없으면,
%         numIRClusters(trial) = 0;
%         %continue;
%     elseif numel(IRs) == 2 % IR이 딱 한번만 끊김
%         numIRClusters(trial) = 1;
%     else
%         IIRI = IRs(2:end,1) - IRs(1:end-1,2);
%         numIRClusters(trial) = sum(IIRI > MIN_IR_CLUSTER_INTERVAL) + 1;
%     end
%         
%     %% Lick Cluster
%     if isempty(Licks) % Lick 이 하나도 없으면,
%         numLickClusters(trial) = 0;
%         %continue;
%     elseif numel(Licks) == 2 % Lick을 딱 한번만 하면
%         numLickClusters(trial) = 1;
%     else
%         ILicksI = Licks(2:end,1) - Licks(1:end-1,2);
%         numLickClusters(trial) = sum(ILicksI > MIN_LICK_CLUSTER_INTERVAL) + 1;
%     end
    
    %% Avoid / Escape
    Trial = ParsedData{trial,1};
    Attack = ParsedData{trial,4};
    if isempty(Attack) % Attack이 없었음. 
        if diff(Trial) > MIN_1MIN_TIMEOUT_DURATION % 1분 시간초과 
            behaviorResult(trial) = 'M';
        else % Give up
            behaviorResult(trial) = 'G';
        end
    else % Attack 이 있었음 : Avoid / Escape 둘 중 하나
        % parameter selection에서 보았듯이 0.5초 이후로 있는 반응은 Attack후 문을 바로 닫지 않아서
        % 생기는 것
        IAttackIROFI = IRs(end,2) - Attack(1); % Attack과 마지막 IR OFF 사이의 시간. +인경우 Escape, -인경우 Avoid
        k = 1;
        while IAttackIROFI > 0.5 % IAttackIROFI가 0.5초 이상이라는 것은 마지막 IR이 Attack 후에 생긴 IR 이라는 의미.
            IAttackIROFI = IRs(end-k,2) - Attack(1);
            k = k + 1;
        end
        clearvars i
        
        if IAttackIROFI >= 0 % Escape
            if SEPARATE_3SEC_6SEC_ESCAPE 
                % true 인 경우 First Lick과 Attack 사이 시간을 사용해서
                % Escape을 C와 D로 구분. 
                % C : Escape on 3sec trial
                % D : Escape on 6sec trial
                if Attack(1) - Licks(1) >= 4.5 % 첫 Lick과 Attack의 사이가 4.5초 이하인경우
                    behaviorResult(trial) = 'C'; % 6초 Escape
                else
                    behaviorResult(trial) = 'D'; % 3초 Escape
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
    behaviorResult(behaviorResult == 'G') = [];
    behaviorResult(behaviorResult == 'M') = [];
end


clearvars Attack Attacks IAttackIROFI IIRI ILicksI IRs k KEEP_ONLY_A_AND_E SEPARATE_3SEC_6SEC_ESCAPE Licks MIN_1MIN_TIMEOUT_DURATION MIN_IR_CLUSTER_INTERVAL MIN_LICK_CLUSTER_INTERVAL numTrial trial Trials Trial
        
end