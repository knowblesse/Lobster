function [ParsedData, Trials, IRs, Licks, Attacks ] = BehavDataParser(targetdir)
%% BehavDataParser
% TDT Open Bridge 에서 추출한 행동 데이터 csv 파일을 불러온다.
% 2018 Knowblesse
%% Constants
DATALIST = {'ATTK', 'ATOF', 'IROF', 'IRON', 'LICK', 'LOFF', 'TROF', 'TRON' }; % parsing에 필요한 데이터의 목록
DATAPAIR = [1,      1,      2,      2,      3,      3,      4,      4      ]; % 에러 확인용 값들. 같은 pair 번호를 가지는 데이터 끼리는 크기가 같아야 함.단, 0인경우 신경 안씀.

%% 폴더 선택
if ~exist('targetdir','var')
    currfolder = uigetdir();
    if currfolder == 0
        error('Error.데이터 파일 폴더를 선택하지 않으셨습니다.','');
    end
    targetdir = currfolder;
else
    currfolder = targetdir;
end

location = strcat(currfolder, '\*.csv');
filelist = ls(location);

%% 분석에 필요한 모든  데이터 csv 파일이 있는지 확인
datafound = zeros(1,numel(DATALIST)); 
filename = cell(numel(DATALIST),1);
for i = 1 : numel(DATALIST) % 이 한줄로된 파일명에 DATALIST에 해당하는 문자가 전부 있는지 확인 
    for j = 1 : size(filelist,1)
        if contains(filelist(j,:),DATALIST{i})
            datafound(i) = 1;
            filename{i} = filelist(j,:);
            break;
        end
    end
end

if sum(datafound) ~= numel(DATALIST) % 파일이 하나라도 없는 경우
    fprintf('%s 경로에\n',location);
    temp = 1:numel(DATALIST);
    nodataindex = temp(not(datafound));
    for i = nodataindex % 없는 놈을 찾아서 error 메시지 생성
        fprintf('%s\n',DATALIST{i});
    end
    fprintf('파일이 누락\n');
    error('csv 파일 누락!','');
end
fprintf('행동 데이터 확인 완료\n');
clearvars i j datafound squeezedFilelist temp nodataindex nodatalist location

%% 행동 데이터 파일 로드
RAWDATA = cell(1,numel(DATALIST));
for i = 1 : numel(DATALIST)
    % 파라미터 지정
    startRow = 2;
    formatSpec = '%*s%s%f%*s%*s%*s%*s%*[^\n\r]';
    % 파일 열기
    fileID = fopen(strcat(currfolder,'\',filename{i}),'r');
    dataArray = textscan(fileID, formatSpec, 'Delimiter', ',', 'TextType', 'string', 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
    fclose(fileID);
    % 제대로 불러왔는지 확인
    if ~strcmp(dataArray{1},DATALIST{i})
        error('Error.\n%s 파일의 데이터 형식이 이상합니다.\n파일 이름과 데이터를 확인하고 데이터가 세 번째 열에 있는지 확인하세요.',DATALIST{i});
    end
    RAWDATA{i} = dataArray{:,2};
end
fprintf('행동 데이터 로드 완료\n');

clearvars startRow formatSpec fileID dataArray filelist filename currfolder ans

%% 이상한거 확인
% 서로 짝이 되는 데이터, 예를 들어 TRON와 TROF가 서로 갯수가 다른 이상한 데이터를 찾아내기 위함.
% 둘둘 짝이 되는 것을 기준으로 작성. 세 가지 파일의 크기가 같은 경우 코드 변경 필요.
PAIR = unique(DATAPAIR);
for i = 1:numel(PAIR)
    if PAIR(i) == 0
        continue;
    end
    j = find(DATAPAIR == PAIR(i));
    if size(RAWDATA{j(1)},1) ~= size(RAWDATA{j(2)},1)
        error('Error.\n%s와 연관된 다른 변수의 크기가 다릅니다.\n데이터 누락의 가능성이 있습니다.\n',DATALIST{j(1)});
    end
end
        
fprintf('데이터 크기 검사 통과.\n');

clearvars i j PAIR

%% 데이터 변환
% 인덱스 확인
TRON_index = find(strcmp(DATALIST,'TRON'));
TROF_index = find(strcmp(DATALIST,'TROF'));
IRON_index = find(strcmp(DATALIST,'IRON'));
IROF_index = find(strcmp(DATALIST,'IROF'));
LICK_index = find(strcmp(DATALIST,'LICK'));
LOFF_index = find(strcmp(DATALIST,'LOFF'));
ATTK_index = find(strcmp(DATALIST,'ATTK'));
ATOF_index = find(strcmp(DATALIST,'ATOF'));
% 데이터 모으기
Trials = [RAWDATA{TRON_index}, RAWDATA{TROF_index}];
IRs = [RAWDATA{IRON_index},RAWDATA{IROF_index}];
Licks = [RAWDATA{LICK_index},RAWDATA{LOFF_index}];
Attacks = [RAWDATA{ATTK_index},RAWDATA{ATOF_index}];
clearvars TRON_index TROF_index IRON_index IROF_index LICK_index LOFF_index ATTK_index ATOF_index
clearvars RAWDATA
% 쪼개기
numTrial = size(Trials,1);
ParsedData = cell(numTrial,4);

% +------------------------+----------------+---------------+----------------+
% | [TRON Time, TROF Time] | [[IRON, IROF]] | [[LICK,LOFF]] | [[ATTK, ATOF]] |
% +------------------------+----------------+---------------+----------------+

%% 시행 단위로 쪼개기
for i = 1 : numTrial
    ParsedData{i,1} = Trials(i,:); % Trial은 그냥 trial 그대로 자름.
    ParsedData{i,2} = IRs(sum(and(IRs>=Trials(i,1), IRs<Trials(i,2)),2) == 2,:) - Trials(i,1);
    ParsedData{i,3} = Licks(sum(and(Licks>=Trials(i,1), Licks<Trials(i,2)),2) == 2,:) - Trials(i,1);
    ParsedData{i,4} = Attacks(sum(and(Attacks>=Trials(i,1), Attacks<Trials(i,2)),2) == 2, :) - Trials(i,1);
end
    
clearvars DATALIST DATAPAIR i 

%clearvars Attacks IRs Licks numTrial Trials

fprintf('%s 폴더 분석 완료\n',targetdir);
end
    
    

