%% 뉴런을 고름
[filename, pathname] = uigetfile('*.mat','Select Neuron .mat file');
if isequal(filename,0) % 선택하지 않은 경우 취소
    return;
end
Paths = strcat(pathname,filename);
if (ischar(Paths))
    Paths = {Paths};
    filename = {filename};
end








%% A2
ts = A2function(Paths{1});
fprintf('A2 Complete\n');

%% EVENT data 경로 선택 및 불러오기
if exist(strcat(pathname,'EVENTS'),'dir') == 7 % 같은 위치에 EVENTS 폴더가 있음
    targetdir = strcat(pathname,'EVENTS');
else
    targetdir = uigetdir('','Select EVENT Folder'); % 같은 위치에 EVENT 폴더가 없으면 사용자에게 물어봄.
    if isequal(targetdir,0)
        return;
    end
end

[TRON, TROF, IRON, IROF, LICK, LOFF, ATTK, ATOF]=GambleRatsBehavParser(targetdir);
[ParsedData, Trials, IRs, Licks, Attacks ] = BehavDataParser(targetdir);
AnalyticValueExtractor; % Avoid Escape 나눔

%% TRON과 TROF를 Avoid냐 Escape냐를 기준으로 나눔
TRON_A = [];
TROF_A = [];
TRON_E = [];
TROF_E = [];

for tr = 1 : numel(TRON) % 모든 trial 에 대해서
    if behaviorResult(tr) == 'A'
        TRON_A = [TRON_A;TRON(tr)];
        TROF_A = [TROF_A;TROF(tr)];
    elseif behaviorResult(tr) == 'E'
        TRON_E = [TRON_E;TRON(tr)];
        TROF_E = [TROF_E;TROF(tr)];
    end
end

%% Avoid에 해당하는 것 먼저 A3를 돌림.
TRON = TRON_A;
TROF = TROF_A;
A3function(ts, TRON, TROF, IRON, IROF, LICK, LOFF, ATTK, ATOF);
        
%% Escape에 해당하는 것을 A3를 돌림.
TRON = TRON_E;
TROF = TROF_E;
A3function(ts, TRON, TROF, IRON, IROF, LICK, LOFF, ATTK, ATOF);
fprintf('Avoid : %d | Escape : %d | Total : %d\n',sum(behaviorResult == 'A'), sum(behaviorResult == 'E'), numel(behaviorResult));   
