function [Neurons, Neuron_names] = loadAlignedData()
%% loadAlignedData
% load Aligned unit data (.mat) and make them into one cell

%% Select Aligned unit data (.mat) path
if exist('TANK_location','var') % For batch script
    cell_location = dir(strcat(TANK_location,'\recording\aligned\*.mat'));
    filename = {cell_location.name};
    pathname = strcat(getfield(cell_location,'folder'),filesep);
else
    global CURRENT_DIR;
    [filename, pathname] = uigetfile(strcat(CURRENT_DIR, '*.mat'), 'Select Unit Data .mat', 'MultiSelect', 'on');
    if isequal(filename,0)
        clearvars filename pathname
        return;
    end
end

Paths = strcat(pathname,filename);
if (ischar(Paths))
    Paths = {Paths};
    filename = {filename};
end

numNeuron = numel(Paths); 

%% Load and make the data into a cell
Neurons = cell(numNeuron,1); 
Neuron_names = cell(numNeuron,1); 
for f = 1 : numNeuron
    load(Paths{f});
    Neurons{f} = Z;
    clearvars Z
    
    t1 = find(filename{f}=='_');
    t2 = filename{f}(1:t1(end)-1);
    Neuron_names{f} = t2;
end
end
