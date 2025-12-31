%% RVn1.csv 에서 location 뽑아내기
% Bridge에서 추출한 RVn1.csv 파일을 Parsing 해서 Location data를 추출

%% Load Data
[filename, pathname] = uigetfile('.csv','위치파일 "*RVn1.csv 파일을 선택해 주세요.');
fileID = fopen([pathname,filename],'r');

formatSpec = '%*s%s%f%d%*d%*d%d';
dataArray = textscan(fileID,formatSpec, 'Delimiter', ',', 'ReturnOnError',false, 'HeaderLines',1);

fclose(fileID);

% EVENT 이름이 RVn1 이 맞는지 확인.
if (dataArray{1}{1} ~= 'RVn1')
    error('RVn1 데이터가 맞나요? 이벤트 이름이 다른것 같습니다.');
end

% 데이터 옮겨담기
TIME = dataArray{2};
CHAN = dataArray{3};
D0 = dataArray{4};

if rem(numel(dataArray{2}),8) ~= 0 
    % 같은 시간에 데이터가 8개 있다는 가정 하에 아래 코드를 진행하기에 에러 체크를 해줍니다.
    error('데이터가 8채널이 아닙니다.')
end

clearvars pathname fileID formatSpec dataArray

%% Parse Data

num_datapoint = numel(TIME) / 8;

% 8개씩 나눠서 데이터를 정렬합니다.
TIME_reshaped = reshape(TIME,8,num_datapoint);
CHAN_reshaped = reshape(CHAN,8,num_datapoint);
D0_reshaped = reshape(D0,8,num_datapoint);

if sum(sum(CHAN_reshaped) ~= 36) ~= 0 
    % 위의 가정을 기반으로 8개씩 자르면 각 자른 조각마다 channel 1,2,3,4,5,6,7,8이 하나씩 있어야 합니다.
    % 항상 87654321 순으로 받으면 정말 좋을텐데 꼭 그런건 아닌가 봅니다.
    % 그래서 채널 값들을 다 합하면 (1+2+3+4+5+6+7+8 = )36이고 이게 아닌 놈들이 행여나 있는지 확인합니다.
    error('채널이 밀렸을 수 있습니다.')
end


% Time extraction
TIME = TIME_reshaped(1,:)';

% 각 데이터 포인트를 짚어가며 채널에 맞는 데이터를 찾아갑니다.
arranged_D0 = zeros(num_datapoint,8);

for i = 1 : num_datapoint
    channelarray = CHAN_reshaped(:,i)';
    for chn = 1 : 8
        arranged_D0(i,channelarray == chn) = D0_reshaped(chn,i);
    end
end

D0 = arranged_D0;
        
clearvars i channelarray *_reshaped chn arranged_D0 filename num_datapoint

Location = {D0(:,3),D0(:,4),D0(:,7),D0(:,8)};

clearvars D0
