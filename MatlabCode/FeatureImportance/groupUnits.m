function [groupingResult, numGroup] = groupUnits(zscoreMatrix, options)
arguments
    zscoreMatrix (:,:) double ;
    options.numCluster = 8;
    options.cutoffLimit = 50;
    options.showGraph = false;
end

Z_ = linkage(zscoreMatrix, 'average', 'correlation');

unitClusterId = cluster(Z_, 'maxclust', options.numCluster);

cnt = histcounts(unitClusterId, 1:options.numCluster+1);
[val, idx] = sort(cnt, 'descend');

numGroup = 0;
groupingResult = zeros(size(zscoreMatrix,1),1);

for clt = 1 : options.numCluster
    if val(clt) >= options.cutoffLimit
        numGroup = numGroup + 1;
        groupingResult(unitClusterId == idx(clt)) = numGroup;
    end
end

if options.showGraph
    fig = figure();
    axes();
    hold on;
    lines = [];
    legends = {};
    for group = 1 : numGroup
        [~, obj_line, ~] = shadeplot(zscoreMatrix(groupingResult == group, :), 'SD', 'sem', 'LineWidth', sum(groupingResult == group)/100);
        lines = [lines, obj_line];
        legends = [legends, strcat("Group : " , num2str(group), " #", num2str(sum(groupingResult == group)))];
    end
    legend(lines, legends);
end
end