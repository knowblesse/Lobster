% Automation
loc0 = 'C:\VCF\Lobster\data\rawdata\';
loc1 = dir(loc0);
loc2 = {loc1.name};
loc2 = loc2(3:end);

for i = 3 : numel(loc2) % 모든 폴더에 대해서
    filename_raw = dir(strcat(loc0,loc2{i},'\aligned*'));
    filename_raw = {filename_raw.name};
    for j = 1 : numel(filename_raw)
        temp = dir(strcat(loc0,loc2{i},'\',filename_raw{j}));
        temp = {temp.name};
        temp = temp(3:end);
        for k = 1 : numel(temp)
            delete(strcat(loc0,loc2{i},'\',filename_raw{j},'\',temp{k}));
        end
        rmdir(strcat(loc0,loc2{i},'\',filename_raw{j}));
    end
end