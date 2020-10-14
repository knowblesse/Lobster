function [ParsedData, Trials, IRs, Licks, Attacks ] = BehavDataParserTank(targetdir)
%% BehavDataParserTank
% Import behavior data from TDT Tank
% 2020 Knowblesse

%% Select Tank
if exist('targetdir','var') <= 0
    targetdir = uigetdir();
    if targetdir == 0
        error('User Cancelled');
    end
end

if exist(targetdir,'dir') <= 0
    error('Tank path %s not found',targetdir);
end

if exist('TDTbin2mat','file') <= 0
    error('Can not find TDTbin2mat function in scope.');
end

DATA = TDTbin2mat(targetdir,'TYPE',{'epocs'});

%% Check data integrity
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
            error('  Can not recover the Last TROF data from BLOF');
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


%% Parse data
% +------------------------+----------------+---------------+----------------+
% | [TRON Time, TROF Time] | [[IRON, IROF]] | [[LICK,LOFF]] | [[ATTK, ATOF]] |
% +------------------------+----------------+---------------+----------------+
Trials = [DATA.epocs.TRON.onset, DATA.epocs.TROF.onset];
IRs = [DATA.epocs.IRON.onset,DATA.epocs.IROF.onset];
Licks = [DATA.epocs.LICK.onset,DATA.epocs.LOFF.onset];
Attacks = [DATA.epocs.ATTK.onset, DATA.epocs.ATOF.onset];

numTrial = size(Trials,1);
ParsedData = cell(numTrial,4);

for i = 1 : numTrial
    ParsedData{i,1} = Trials(i,:);
    ParsedData{i,2} = IRs(sum(and(IRs>=Trials(i,1), IRs<Trials(i,2)),2) == 2,:) - Trials(i,1);
    ParsedData{i,3} = Licks(sum(and(Licks>=Trials(i,1), Licks<Trials(i,2)),2) == 2,:) - Trials(i,1);
    ParsedData{i,4} = Attacks(sum(and(Attacks>=Trials(i,1), Attacks<Trials(i,2)),2) == 2, :) - Trials(i,1);
end

fprintf('%s : Complete\n',DATA.info.blockname);
end
    
    

