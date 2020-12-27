function [ParsedData, Trials, IRs, Licks, Attacks ] = BehavDataParser(targetdir)
%% BehavDataParser
% Import Event data from 1) Open Bridge extracted csv files or 2) Tank
% Invalid trial is automatically removed and the remaining is concatenated.
% Thus resulting trial number might differ from the original number.
% Created on 2018 Knowblesse
% Modified on 2020NOV19 Knowblesse
%% Constants

%% Select folder
if exist('targetdir','var') <= 0
    global CURRENT_DIR;
    if ~isempty(CURRENT_DIR)
        targetdir = uigetdir(CURRENT_DIR);
    else
        targetdir = uigetdir();
    end
    if targetdir == 0
        error('User Cancelled');
    end
end
%% Check if it is a Tank
if exist(strcat(targetdir,'\StoresListing.txt'))
    % the file is a tank
    if exist(targetdir,'dir') <= 0
        error('Tank path %s not found',targetdir);
    end
    if exist('TDTbin2mat','file') <= 0
        error('Can not find TDTbin2mat function in scope.');
    end
    fprintf('Tank Found');
    isTank = true;
else
    % the file is a folder with .csv s
    DATALIST = {'ATTK', 'ATOF', 'IROF', 'IRON', 'LICK', 'LOFF', 'TROF', 'TRON' }; % Events
    DATAPAIR = [1,      1,      2,      2,      3,      3,      4,      4      ]; % Event paring marker. Same numbered event should have same number of data. 0 means skip the data length checking
    location = strcat(targetdir, '\*.csv');
    filelist = ls(location);
    % Check all csv files are present
    datafound = zeros(1,numel(DATALIST)); 
    filename = cell(numel(DATALIST),1);
    for i = 1 : numel(DATALIST) % check all file names whether they have all DATALIST strings
        for j = 1 : size(filelist,1)
            if contains(filelist(j,:),DATALIST{i})
                datafound(i) = 1;
                filename{i} = filelist(j,:);
                break;
            end
        end
    end

    if sum(datafound) ~= numel(DATALIST) % if there is a missing file
        fprintf('Path : %s \n',location);
        temp = 1:numel(DATALIST);
        nodataindex = temp(not(datafound));
        for i = nodataindex % find the name of the missing file
            fprintf('%s\n',DATALIST{i});
        end
        fprintf('file is missing\n');
        error('csv file missing!');
    end
    fprintf('CSV folder found');
    clearvars location filelist datafound i j temp nodataindex 
    isTank = false;
end
fprintf('Event Folder found\n');

%% Load Event Data
if isTank
    DATA = TDTbin2mat(targetdir,'TYPE',{'epocs'});
else
    RAWDATA = cell(1,numel(DATALIST));
    for i = 1 : numel(DATALIST)
        startRow = 2;
        formatSpec = '%*s%s%f%*s%*s%*s%*s%*[^\n\r]';
        fileID = fopen(strcat(targetdir,'\',filename{i}),'r');
        dataArray = textscan(fileID, formatSpec, 'Delimiter', ',', 'TextType', 'string', 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
        fclose(fileID);
        % check column sequence
        if ~strcmp(dataArray{1},DATALIST{i})
            error('Error.\n%s Check column sequence. Something is different.',DATALIST{i});
        end
        RAWDATA{i} = dataArray{:,2};
    end
    clearvars startRow formatSpec fileID dataArray filelist filename
end
fprintf('Event data loaded\n');



%% Check data integrity
% check number of datapoints of paired events (ex. TRON - TROF)
% check all events are loaded (Tank folder only)
if isTank
    % Field check
    Fields = {'BLON','BLOF','TRON','TROF','IRON','IROF','LICK','LOFF','ATTK','ATOF'};
    for f = Fields
        if ~isfield(DATA.epocs,f)
            warning('%s : %s does not exist!!',cell2mat(f), DATA.info.blockname);
        end
    end
    clearvars f Fields
    
    % Number check
    if size(DATA.epocs.TRON.onset,1) ~= size(DATA.epocs.TROF.onset,1)
        warning('%s : TRON TROF size mismatch!!',DATA.info.blockname);
        if size(DATA.epocs.TRON.onset,1) == size(DATA.epocs.TROF.onset,1) + 1
            if isfield(DATA.epocs,'BLOF')
                warning('  Possibly the last TROF skipped. Using BLOF instead.');
                DATA.epocs.TROF.onset = [DATA.epocs.TROF.onset;DATA.epocs.BLOF.onset];
            else
                warning('  Can not recover the Last TROF data from BLOF\n  Deleting the last trial');
                DATA.epocs.TRON.onset = DATA.epocs.TRON.onset(1:end-1);
            end
        else
            error('  Critical Error');
        end
    end
    if size(DATA.epocs.IRON.onset,1) ~= size(DATA.epocs.IROF.onset,1)
        error('%s : IRON IROF size mismatch!!',DATA.info.blockname);
    end
    if size(DATA.epocs.LICK.onset,1) ~= size(DATA.epocs.LOFF.onset,1)
        error('%s : LICK LOFF size mismatch!!',DATA.info.blockname);
    end
else
    % number check
    PAIR = unique(DATAPAIR);
    for i = 1:numel(PAIR)
        if PAIR(i) == 0
            continue;
        end
        j = find(DATAPAIR == PAIR(i));
        if size(RAWDATA{j(1)},1) ~= size(RAWDATA{j(2)},1)
            error('Error.\n%s size mismatch!!n',DATALIST{j(1)});
        end
    end
    clearvars i j PAIR
end    
fprintf('Data integrity test passed.\n');

%% Parse data
if isTank
    Trials = [DATA.epocs.TRON.onset, DATA.epocs.TROF.onset];
    IRs = [DATA.epocs.IRON.onset,DATA.epocs.IROF.onset];
    Licks = [DATA.epocs.LICK.onset,DATA.epocs.LOFF.onset];
    Attacks = [DATA.epocs.ATTK.onset, DATA.epocs.ATOF.onset];
    dataname = DATA.info.blockname;
    clearvars DATA
else
    Trials = [RAWDATA{find(strcmp(DATALIST,'TRON'))}, RAWDATA{find(strcmp(DATALIST,'TROF'))}];
    IRs = [RAWDATA{find(strcmp(DATALIST,'IRON'))},RAWDATA{find(strcmp(DATALIST,'IROF'))}];
    Licks = [RAWDATA{find(strcmp(DATALIST,'LICK'))},RAWDATA{find(strcmp(DATALIST,'LOFF'))}];
    Attacks = [RAWDATA{find(strcmp(DATALIST,'ATTK'))},RAWDATA{find(strcmp(DATALIST,'ATOF'))}];
    dataname = targetdir;
    clearvars RAWDATA
end


% Generate Variable ParsedData
% +------------------------+----------------+---------------+----------------+
% | [TRON Time, TROF Time] | [[IRON, IROF]] | [[LICK,LOFF]] | [[ATTK, ATOF]] |
% +------------------------+----------------+---------------+----------------+


numTrial = size(Trials,1);
ParsedData = cell(numTrial,4);

for i = 1 : numTrial
    ParsedData{i,1} = Trials(i,:);
    ParsedData{i,2} = IRs(sum(and(IRs>=Trials(i,1), IRs<Trials(i,2)),2) == 2,:) - Trials(i,1);
    ParsedData{i,3} = Licks(sum(and(Licks>=Trials(i,1), Licks<Trials(i,2)),2) == 2,:) - Trials(i,1);
    ParsedData{i,4} = Attacks(sum(and(Attacks>=Trials(i,1), Attacks<Trials(i,2)),2) == 2, :) - Trials(i,1);
end

%% Remove invalid trials (no IRON or no LICK or no ATTK)
numTrial = size(ParsedData,1);
validtrial = false(numTrial,1);
for t = 1 : numTrial
    if isempty(ParsedData{t,2})
        warning('Trial %d is invalid : No IR => Removed',t);
    elseif isempty(ParsedData{t,3})
        warning('Trial %d is invalid : No Lick => Removed',t);
    elseif isempty(ParsedData{t,4})
        warning('Trial %d is invalid : No Attk => Removed',t);
    else
        validtrial(t) =  true;
    end
end

ParsedData = ParsedData(validtrial, :);
fprintf('%s : Complete\n',dataname);
