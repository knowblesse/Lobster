%% EventDecoder_AW_timing
% Relationship between Event Decoder's accuracy vs Avoidance Withdrawal
% timing
addpath('../Behavior/');

%%
BNBResult = load('D:\Data\Lobster\BNB_Result_fullshuffle.mat');
Behavior_Data_path = 'D:\Data\Lobster\BehaviorData';

output = [];
for i_session = 1 : 40
    i_session = 19;
    val = regexp(BNBResult.tankNames(i_session, :), '(?<tankName>#.*L)', 'names');
    behavior_data = load(fullfile(Behavior_Data_path, strcat(val.tankName, '.mat')));
    
    % Check numTrial with Data
    if size(behavior_data.ParsedData, 1) ~= ...
            size(BNBResult.result{i_session}.WholeTestResult_HWAE, 1)
        error("Size does not match");
    end
    
    %     
    [behaviorResult, ParsedData] = analyticValueExtractor(behavior_data.ParsedData, false, false);
    
    behaviorResult_index = 1;
    for i_trial = 1 : numel(behaviorResult)
        if behaviorResult(i_trial) == 'A'
            IR = ParsedData{i_trial, 2};
            Attack = ParsedData{i_trial, 4};
            nearAttackIRindex = find(IR(:,1) < Attack(1),1,'last');
            HW_timing = IR(nearAttackIRindex, 2);
            
            % Decoder Result
            % First Column is true. 0 is A
            sum((BNBResult.result{i_session}.WholeTestResult_HWAE(:,2) >= 0.5) == BNBResult.result{i_session}.WholeTestResult_HWAE(:,1)) / 82
            behaviorResult_index = behaviorResult_index + 1;
            
            
    
    