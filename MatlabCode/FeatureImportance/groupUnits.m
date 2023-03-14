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
    if numGroup == 2
        colors_bw = [0, 0, 0; 0.5, 0.5, 0.5];
    elseif numGroup == 3
        colors_bw = [0, 0, 0; 0.4, 0.4, 0.4; 0.8, 0.8, 0.8];
    end
    fig = figure();
    axes();
    hold on;
    lines = [];
    legends = {};
    for group = 1 : numGroup
        [~, obj_line, ~] = shadeplot(...
            zscoreMatrix(groupingResult == group, :),...
            'SD', 'sem',... %'LineWidth', sum(groupingResult == group)/100,...
            'LineWidth', 1.3,...
            'FaceAlpha', 0.3,...
            'Color', colors_bw(group,:));
        lines = [lines, obj_line];
        legends = [legends, strcat("Group " , num2str(group))];
    end
    line(xlim, [0,0], 'LineStyle', ':', 'Color', [0.3, 0.3, 0.3]);
    ylabel('Z score');
    xlabel('Time (sec)');
    xticks(0:20:80);
    xticklabels(-1:0.5:1);
    legend(lines, legends, 'FontSize', 6.6);
    set(gca, 'FontName', 'Noto Sans');
    pos = get(gcf, 'Position');
    set(gcf, 'Position', [pos(1), pos(2), 288, 236]);

    %% Create Dendrogram
    figure();
    
    H = dendrogram(Z_, 632, 'ColorThreshold', Z_(end-8, 3));
    lineColors = zeros(632,3);
    for i = 1 : 631
        lineColors(i,:) = H(i).Color;
    end

    for i = 1 : 631
        if sum(all(H(i).Color == lineColors, 2)) < 50
            H(i).Color = [1, 1, 1];
        elseif numGroup == 2 % stupid hard coding
            if sum(all(H(i).Color == lineColors, 2)) > 300
                H(i).Color = [0, 0, 0];
            else
                H(i).Color = [0.5, 0.5, 0.5];
            end
        elseif numGroup == 3
            if sum(all(H(i).Color == lineColors, 2)) > 300
                H(i).Color = [0, 0, 0];
            elseif sum(all(H(i).Color == lineColors, 2)) > 100
                H(i).Color = [0.4, 0.4, 0.4];
            else
                H(i).Color = [0.8, 0.8, 0.8]; 
            end
        end
    end
    xticks([]);
    xlabel('Cells');
    ylabel('Distance (1-r)');
    set(gca, 'FontName', 'Noto Sans');
    pos = get(gcf, 'Position');
    set(gcf, 'Position', [pos(1), pos(2), 288, 236]);
end
end
