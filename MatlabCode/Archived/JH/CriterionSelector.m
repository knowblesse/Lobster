%% CriterionSelector
[Neurons, Neuron_names] = loadAlignedData();
numNeuron = numel(Neurons);
%% Constants
minimumFR = 0.2;
minimumZ = 2;

%% MakeList
% Col1 : 걍 번호
% Col2 : 뉴런명
% Col3 : 눈으로 고른 뉴런여부
% Col4 : 기준으로 고른 뉴런여부
% Col5 : IRON에 리스폰시브 
% Col6 : IROF에 리스폰시브
list = cell(numNeuron,3);
for i = 1 : numNeuron    
    list{i,1} = i;
    list{i,2} = Neuron_names;
    list{i,3} = 0;
    list{i,4} = 0;
    list{i,5} = 0;
    list{i,6} = 0;
end
clearvars t1 t2

%% Mark Handpicked data
for d = 1 : numel(data{1})
    for n = 1 : numel(data{2}{d})
        nnn = strcat('0',num2str(data{1}(d)),'_',num2str(data{2}{d}(n)));
        for i = 1 : numNeuron
            if strcmp(list{i,2},nnn)
                list{i,3} = 1;
            end
        end
    end
end

clearvars nnn d n 

%% Mark Criterion Selected data
for i = 1 : numNeuron
    if CheckList(i,1) > minimumFR
        if CheckList(i,2) > minimumZ % is IRON responsive 
            list{i,4} = 1;
            list{i,5} = 1;
        end
        if CheckList(i,3) > minimumZ % is IROF responsive
            list{i,4} = 1;
            list{i,6} = 1;
        end
    end
end

% Unmark false wave form data
for i = 1 : numel(delete_ind)
    list{delete_ind(i),4} = 0;
end

%% HMFC Hit Miss False alarm Correct rejection
% Hit Miss
% FA  CR
HMFC = zeros(2,2);
for i = 1: numNeuron
    if and(list{i,3} == 1,list{i,4} == 1)
        HMFC(1,1) = HMFC(1,1) + 1;
    elseif and(list{i,3} == 0, list{i,4} == 0)
        HMFC(2,2) = HMFC(2,2) + 1;
    elseif and(list{i,3} == 1, list{i,4} == 0)
        HMFC(1,2) = HMFC(1,2) + 1;
    elseif and(list{i,3} == 0, list{i,4} == 1)
        HMFC(2,1) = HMFC(2,1) + 1;
    else
        warning('????');
    end
end

clearvars minimum*


%% Print M F
fprintf('Miss **************** \n');
ind = 1;
for i = 1 : numNeuron
    if and(list{i,3} == 1, list{i,4} == 0)
        fprintf('%2d : ',ind);
        fprintf(list{i,2});
        fprintf('\n');
        ind = ind + 1;
    end
end
fprintf('False Alarm **************** \n');
ind = 1;
for i = 1 : numNeuron
    if and(list{i,3} == 0, list{i,4} == 1)
        fprintf('%2d : ',ind);
        fprintf(list{i,2});
        fprintf('\n');
        ind = ind + 1;
    end
end

disp(HMFC);

