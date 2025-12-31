%% SessionInfoExtractor
% Analyze multiple session Tank folders and print information.

%% Select Session Tanks
global CURRENT_DIR
if ~isempty(CURRENT_DIR)
    targetdir = uigetdir(CURRENT_DIR);
else
    targetdir = uigetdir();
end
if targetdir == 0
    error('SessionInfoExtractor : User Cancelled');
end

%% Analyze
[ParsedData, ~, ~, ~, ~, targetdir] = BehavDataParser(targetdir);

tankName = regexp(targetdir, '.*(\\.)*\\(?<tank_name>.*)', 'names');
tankName = tankName.tank_name;
numTrial = size(ParsedData,1);
numLick = size(cell2mat(ParsedData(:,3)),1);

%% Session time
sTime = ParsedData{end,1}(end) - ParsedData{1,1}(1);

%% A/E ratio
if isempty(ParsedData{1,4}) 
    % Training ( Shuttling )
    numAvoid = 0;
    numEscape = 0;
    for trial = 1 : size(ParsedData,1)
        doorCloseTime = ParsedData{trial, 3}(1) + 6;
        nearAttackIRindex = find(ParsedData{trial, 2}(:,1) < doorCloseTime,1,'last'); % Shuttling 6 seconds
        IAttackIROFI = ParsedData{trial, 2}(nearAttackIRindex, 2) - doorCloseTime;
        if IAttackIROFI >= 0 % Escape
            numEscape = numEscape + 1;
        else % Avoid
            numAvoid = numAvoid + 1;
        end
    end    
else
    behaviorResult = analyticValueExtractor(ParsedData, false, false);
    numAvoid = sum(behaviorResult == 'A');
    numEscape = sum(behaviorResult == 'E');
end

fprintf('*****************************************************************\n');
fprintf('Tank : %s\n', tankName);
fprintf('Duration : %s\n', duration(seconds(sTime),'Format','mm:ss'))
fprintf('|----------+---------+----------+-----------|\n');
fprintf('| numTrial | numLick | numAvoid | numEscape |\n');
fprintf('| %8d | %7d | %8d | %9d |\n', numTrial, numLick, numAvoid, numEscape);
fprintf('|----------+---------+----------+-----------|\n');
   
