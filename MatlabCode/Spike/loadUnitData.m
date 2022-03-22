function [Paths, pathname, filename] = loadUnitData(TANK_location)
%% loadUnitData
% load unit data (.mat) into path variable. 
% Created by Knowblesse 20DEC04

%% Select Unit data (.mat) path
if exist('TANK_location','var') % For batch script
    cell_location = dir(strcat(TANK_location,'\recording\*.mat'));
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
fprintf('loadUnit : %d units loaded\n',numel(Paths));
end