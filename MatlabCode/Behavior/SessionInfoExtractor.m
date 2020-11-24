%% SessionExtractor
% Create a information matrix about total number of avoid and escape trial
% in each sessions.

loc0 = 'C:\VCF\Lobster\data\rawdata\';
loc1 = dir(loc0);
loc2 = {loc1.name};
loc2 = loc2(3:end);

SessionData = zeros(numel(loc2),3);
% [Avoid | Escape | isSuc] 
for i = 1 : numel(loc2) % 모든 폴더에 대해서
    pathname = strcat(loc0,loc2{i},'\');
    %% 이벤트 파일 분석
    targetdir = strcat(pathname,'EVENTS');
    [ParsedData, ~, ~, ~, ~] = BehavDataParser(targetdir);
    
    if contains(loc2{i}, 'suc') % suc 데이터 라면
        SessionData(i,3) = 1;
        SessionData(i,1) = size(ParsedData,1);
    else % suc 데이터가 아니라면
        behaviorResult = AnalyticValueExtractor(ParsedData,false,false);
        SessionData(i,3) = 0;
        SessionData(i,1) = sum(behaviorResult == 'A');
        SessionData(i,2) = sum(behaviorResult == 'E');
    end
end
    
    
    
    
