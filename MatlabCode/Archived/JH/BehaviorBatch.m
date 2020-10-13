%% Behavior Analysis Batch
% Behavior 데이터를 한번에 분석. 

% Automation
loc0 = 'C:\VCF\Lobster\data\rawdata\';
loc1 = dir(loc0);
loc2 = {loc1.name};
loc2 = loc2(3:end);

% Data Storage
Behav = struct();
t = 1;

for i = 1 : numel(loc2) % 모든 폴더에 대해서
    if contains(loc2{i}, 'suc') % suc 데이터 라면
        continue;
    end
    
    % 현재 경로 설정
    pathname = strcat(loc0,loc2{i},'\');
    
    %% EVENT data 경로 선택 및 불러오기
    if exist(strcat(pathname,'EVENTS'),'dir') == 7 % 같은 위치에 EVENTS 폴더가 있음
        targetdir = strcat(pathname,'EVENTS');
    else
        error('EVENT 폴더 누락');
    end

    %% BehavDataParser를 돌림.
    [ParsedData, Trials, IRs, Licks, Attacks ] = BehavDataParser(targetdir);
    
    clearvars targetdir;
    
    %% Avoid / Escape 데이터를 추출.
    behaviorResult = AnalyticValueExtractor(ParsedData,false,false);
    
    %% Struct 구조로 데이터를 출력. 
    Behav(t).name = loc2{i};
    Behav(t).ParsedData = ParsedData;
    Behav(t).Trials = Trials;
    Behav(t).IRs = IRs;
    Behav(t).Licks = Licks;
    Behav(t).Attacks = Attacks;
    Behav(t).numA = sum(behaviorResult == 'A');
    Behav(t).numE = sum(behaviorResult == 'E');
    t = t + 1;
    
    
    clearvars -except loc0 loc1 loc2 i t Behav;
end