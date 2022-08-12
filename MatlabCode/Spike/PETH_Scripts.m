%% PETH_Scripts
output = loadAllUnitData();

%% Draw Single PETH
index = 15 ;
event = 'valid_IROF'; % ["TRON","first_IRON","valid_IRON","first_LICK","valid_IROF","ATTK","TROF"]
TIMEWINDOW = [-1000, 1000];

timepoint = getTimepointFromParsedData(output.BehavData{index});

% Generate spikes cell to feed in to the drawPETH function
numTrial = size(timepoint.TRON, 1);
spikes = cell(numTrial,1);
for trial = 1 : numTrial
    spikes{trial} = output.RawSpikeData{index}(...
        and(output.RawSpikeData{index} >= timepoint.(event)(trial) + TIMEWINDOW(1), output.RawSpikeData{index} < timepoint.(event)(trial) + TIMEWINDOW(2)))...
        - timepoint.(event)(trial);
end

fig = figure(...
    'Name', strcat("Index : ", num2str(index)," event : ", event),...
    'Position', [1094, 592, 560, 301]...
    );
clf;
ax_raster1 = subplot(3,1,1:2);
title(strcat(output.Session{index}, '-', num2str(output.Cell(index))), 'Interpreter', 'none');
ax_histo1 = subplot(3,1,3);
drawPETH(spikes, TIMEWINDOW, ax_raster1, ax_histo1, false);
clearvars -except output

%% Draw Representative A/E HW PETH
TIMEWINDOW = [-1000, 1000];
for index = [72, 516]
    %nLoad BehaviorData and Load Timepoints
    numTrial = size(output.Data{index}.binned_spike.TRON, 1);
    ParsedData = output.BehavData{index};
    behaviorResult = analyticValueExtractor(ParsedData, false, true);
    timepoint = getTimepointFromParsedData(ParsedData);

    % Divide Avoid and Escape PETH
    event = 'valid_IROF';
    spikes = cell(numTrial,1);
    for trial = 1 : numTrial
        spikes{trial} = output.RawSpikeData{index}(...
            and(output.RawSpikeData{index} >= timepoint.(event)(trial) + TIMEWINDOW(1), output.RawSpikeData{index} < timepoint.(event)(trial) + TIMEWINDOW(2)))...
            - timepoint.(event)(trial);
    end

    fig = figure(...
        'Name', strcat("Index : ", num2str(index), " event : Head Withdrawal"),...
        'Position', [1094, 592, 560, 301]);
    clf;
    ax_raster1 = subplot(3,2,[1,3]);
    title("Avoid", 'FontName', 'Noto Sans');
    ax_histo1 = subplot(3,2,5);
    behaviorIndex = behaviorResult == 'A';
    drawPETH(spikes(behaviorIndex, :), TIMEWINDOW, ax_raster1, ax_histo1, true);

    ax_raster2 = subplot(3,2,[2,4]);
    title("Escape", 'FontName', 'Noto Sans');
    ax_histo2 = subplot(3,2,6);
    behaviorIndex = behaviorResult == 'E';
    drawPETH(spikes(behaviorIndex, :), TIMEWINDOW, ax_raster2, ax_histo2, true);
    
    yl1 = ylim(ax_histo1);
    yl2 = ylim(ax_histo2);
    
    maxlim = max(yl1(2), yl2(2));
    ylim(ax_histo1, [0, maxlim]);
    ylim(ax_histo2, [0, maxlim]);
end
clearvars -except output


%% Responsiveness calculation
zscore_threshold = 4;
valid_IRON_zscores = zeros(size(output,1), 40);
valid_IROF_zscores = zeros(size(output,1), 40);
valid_IROF_A_zscores = zeros(size(output,1), 40);
valid_IROF_E_zscores = zeros(size(output,1), 40);
responsive_IRON = zeros(size(output,1),1);
responsive_IROF = zeros(size(output,1),3);
for i = 1 : size(output, 1)        
    valid_IRON_zscores(i, :) = output.Zscore{i}.valid_IRON;
    valid_IROF_zscores(i, :) = output.Zscore{i}.valid_IROF;
    valid_IROF_A_zscores(i, :) = output.Zscore{i}.valid_IROF_A;
    valid_IROF_E_zscores(i, :) = output.Zscore{i}.valid_IROF_E;
    
    responsive_IRON(i) = any(abs(valid_IRON_zscores(i, :)) > zscore_threshold);
    
    responsive_IROF(i,1) = any(abs(valid_IROF_zscores(i, :)) > zscore_threshold);
    responsive_IROF(i,2) = any(abs(valid_IROF_A_zscores(i, :)) > zscore_threshold);
    responsive_IROF(i,3) = any(abs(valid_IROF_E_zscores(i, :)) > zscore_threshold);
end

fprintf('HE Responsive unit : %.2f %%\n', sum(responsive_IRON) / size(output,1)*100);
fprintf('HW Responsive unit : %.2f %%\n', sum(responsive_IROF(:,1)) / size(output,1)*100);
fprintf('   Avoid Responsive : %.2f %% Escape Responsive : %.2f %%\n\n', sum(responsive_IROF(:,2)) / size(output,1)*100, sum(responsive_IROF(:,3)) / size(output,1)*100);

clearvars -except output valid* responsive*
