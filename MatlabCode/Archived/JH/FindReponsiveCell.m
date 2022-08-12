%% FindResponsiveCell
% Go though all neurons and find the responsive cells
% Prerequisite
%   - All Session files must gone through AlignEvent.m script and have aligned data

basepath = 'F:\LobsterData';

filelist = dir(basepath);
workingfile = regexp({filelist.name},'^#\S*','match');
workingfile = workingfile(~cellfun('isempty',workingfile));

result = array2table(zeros(0,4),'VariableNames', ["Session", "Cell Number", "Approach", "Avoidance"]);
for f = 1 : numel(workingfile)
    TANK_location = strcat(basepath,filesep, cell2mat(workingfile{f}));
    matfileList = dir(strcat(TANK_location, filesep, 'recording\aligned'));
    matfileList = regexp({matfileList.name},'\S*mat','match');
    matfileList = matfileList(~cellfun('isempty', matfileList));
    for mf = 1 : numel(matfileList)
        load(strcat(TANK_location, filesep, 'recording\aligned\',cell2mat(matfileList{mf})));
        result = [result; {workingfile{f}, mf,...
            any(abs(Z.zscore.valid_IRON)>3),...
            any(abs(Z.zscore.valid_IROF)>3)}...
            ];
    end
end
fprintf('DONE\n');
