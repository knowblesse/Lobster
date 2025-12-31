function output = getMeanDevPath(path_shut3, path_lob1)
%% getMeanDevPath
% get mean of deviation from the center row line of the Path.
% When the path from the nest to the encounter zone is not straight, 
% then this value increases.
% Main purpose of this script is to measure the change of behavior between shut 3 and lob 1

CENTER_ROW = 280;

%% Load Data
data_shut3 = readmatrix(path_shut3);
data_lob1 = readmatrix(path_lob1);

%% Find Approach lines
output = [];
for data = {data_shut3, data_lob1}
    data = data{1};

    isOutDone = true;
    isInNest = false;
    isOut = false;
    lastStartIndex = 0;
    startIndex = [];
    endIndex = [];
    for i = 1 : size(data,1)
        if data(i, 3) < 250 & (data(i,2) > 230 & data(i,2) < 320) 
            isOut = false;
            isOutDone = false;
            isInNest = true;
            lastStartIndex = i;
        end

        if isInNest & data(i,3) > 265
            isOut = true;
            isInNest = false;
        end

        if isOut & data(i,3) > 500
            isOutDone = true;
            isOut = false;
            startIndex = [startIndex; lastStartIndex];
            endIndex = [endIndex; i];
        end
    end

    % For every approach lines, 
    outputMatrix = zeros(size(endIndex,1),1);

    for i = 1 : size(endIndex,1)
        outputMatrix(i) = mean(abs(data(startIndex(i):endIndex(i), 2) - CENTER_ROW));
    end
    output = [output, mean(outputMatrix)];
end
end

