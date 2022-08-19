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


