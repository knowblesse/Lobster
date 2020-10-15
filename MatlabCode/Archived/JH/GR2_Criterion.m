%% CriterionSelector
[Neurons, Neuron_names] = loadAlignedData();
numNeuron = numel(Neurons);


%% Get fr and z data
CheckList = zeros(numNeuron,3);
for f = 1 : numNeuron
    CheckList(f,1) = Neurons{f}.FR_trial;
    CheckList(f,2) = max(abs(Neurons{f}.IRON));
    CheckList(f,3) = max(abs(Neurons{f}.IROF));
end

%% Constants
minimumFR = 0.2;
minimumZ = 2;

%% MakeList
% Col1 : 걍 번호
% Col2 : 뉴런명
% Col3 : 기준으로 고른 뉴런여부
% Col4 : IRON에 리스폰시브 
% Col5 : IROF에 리스폰시브
list = cell(numNeuron,3);
for i = 1 : numNeuron    
    list{i,1} = i;
    list{i,2} = Neuron_names;
    list{i,3} = 0;
    list{i,4} = 0;
    list{i,5} = 0;
end
clearvars t1 t2

%% Mark Criterion Selected data
for i = 1 : numNeuron
    if CheckList(i,1) > minimumFR
        if CheckList(i,2) > minimumZ % is IRON responsive 
            list{i,3} = 1;
            list{i,4} = 1;
        end
        if CheckList(i,3) > minimumZ % is IROF responsive
            list{i,3} = 1;
            list{i,5} = 1;
        end
    end
end
