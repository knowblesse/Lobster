function [firedIndices] = firedLocationParser(result)
%% spike time 데이터가 있는 txt 파일을 읽어서 위치와 연결해준다.
%   @param result : locationParser가 뱉어내는 result 파일.
%       이 파일의 첫번째 열 데이터를 토대로 index들을 뱉어낸다.

%% Load Data
[filename, pathname] = uigetfile('.txt');
fileID = fopen([pathname,filename],'r');

formatSpec = '%f%[^\n]';
spikes = textscan(fileID,formatSpec, 'Delimiter', ',', 'ReturnOnError',false);
spikes = spikes{1};

fclose(fileID);

clear filename pathname fileID formatSpec

%% Find Indexes
numSpike = size(spikes,1);
spikeIndex = zeros(numSpike,1);

% spike 데이터는 spike가 나타났던 시간 데이터 밖에 존재하지 않음.
% 그러므로, 해당 timestamp 와 가장 가까운 location 데이터를 연결해줌
for i = 1 : numSpike
    difference = abs(result(:,1) - spikes(i));
    [~,spikeIndex(i)] = min(difference);
end

firedIndices = spikeIndex;

end
    
