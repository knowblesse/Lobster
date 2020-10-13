% runthis 의 automation

% Automation
loc0 = 'C:\VCF\Lobster\data\rawdata\';
loc1 = dir(loc0);
loc2 = {loc1.name};
loc2 = loc2(3:end);

%% Save Data Location
savepath = 'C:\VCF\Lobster\data\processedData\NeuronProfile\';

for i = 1 : numel(loc2) % 모든 폴더에 대해서
    if ~contains(loc2{i}, 'suc') % suc 데이터 라면 안돌림
        %% Neuron.mat 파일리스트 자동 생성
        filename_raw = dir(strcat(loc0,loc2{i},'\*.mat'));
        filename_raw = {filename_raw.name};
        filename = filename_raw(1:end-1); %UNIT 파일 제외

        pathname = strcat(loc0,loc2{i},'\');

        Paths = strcat(pathname,filename);
        if (ischar(Paths))
            Paths = {Paths};
            filename = {filename};
        end
        %% 각 Unit data 별로 timewindow에 들어가는 데이터를 뽑아냄.
        for f = 1 : numel(Paths) % 선택한 각각의 Unit Data에 대해서...
            %% parse filename
            filename_date = filename{f}(strfind(filename{1},'GR7-')+6:strfind(filename{1},'GR7-')+9);
            temp1 = strfind(filename{f},'_');
            temp2 = strfind(filename{f},'.mat');
            filename_cellnum = filename{f}(temp1(end)+1:temp2-1);

            filename_ = strcat(filename_date,'_',filename_cellnum);
            
            %% Run
            
            %% A2
            ts = A2function(Paths{f},strcat(savepath,filename_));
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
            A3function(ts, TRON, TROF, IRON, IROF, LICK, LOFF, ATTK, ATOF,strcat(savepath,filename_,'_2.png'));

            %% Escape에 해당하는 것을 A3를 돌림.
            TRON = TRON_E;
            TROF = TROF_E;
            A3function(ts, TRON, TROF, IRON, IROF, LICK, LOFF, ATTK, ATOF,strcat(savepath,filename_,'_3.png'));
            fprintf('Avoid : %d | Escape : %d | Total : %d\n',sum(behaviorResult == 'A'), sum(behaviorResult == 'E'), numel(behaviorResult));   
            

            clearvars -except loc0 loc1 loc2 i savepath f Paths filename pathname;
        end
    end
end