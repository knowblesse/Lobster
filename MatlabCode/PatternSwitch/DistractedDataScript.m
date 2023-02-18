%% DistractedDataScript
% Script for analyzing hand-labeled distracted data
% Warning : the bool_distracted only contains values inside the nest zone.

px2cm = 0.169;
truncatedTimes_s = 10;
neural_data_rate = 20;

numTank = 21;
load('tankList.mat');
output_table = table(strings(numTank, 1), zeros(numTank, 1), zeros(numTank, 1), zeros(numTank, 2), zeros(numTank, 1), ...
    'VariableNames', {'Tank', 'Error_Distracted', 'Error_Engaged', 'NumData', 'DataRatio'});

%% 
for i = 1 : numTank
    tank_name = tank_list(i);
    load("D:\Data\Lobster\FineDistanceDataset\" + tank_name + "\" + tank_name + "_frameInfo.mat");
    load("D:\Data\Lobster\FineDistanceResult_syncFixed\" + tank_name + "result_distance.mat");
    distracted = readmatrix("E:\Data\" + tank_name + "\bool_distracted.csv");
    
    % Cure weird frame number
    wrong_frame_num_index = find(diff(frameNumber) < 0);
    frameNumber(wrong_frame_num_index) = round((frameNumber(wrong_frame_num_index-1) + frameNumber(wrong_frame_num_index+1)) / 2);

    midPointTimes = truncatedTimes_s + (1/neural_data_rate)*(0:size(WholeTestResult,1)-1) + 0.5 * (1/neural_data_rate);
    
    % Construct Frame Number correction
    neuralData_isDistracted = logical(distracted(round(interp1(frameTime, frameNumber, midPointTimes))));
    isNesting = WholeTestResult(:,2) < 225;

    % Compare Error btw shuffled and predicted
    output_table.Tank(i) = tank_name;
    output_table.Error_Distracted(i) = mean(abs(WholeTestResult(neuralData_isDistracted & isNesting, 3) - WholeTestResult(neuralData_isDistracted & isNesting, 5)) * px2cm);
    output_table.Error_Engaged(i) = mean(abs(WholeTestResult((~neuralData_isDistracted) & isNesting, 3) - WholeTestResult((~neuralData_isDistracted) & isNesting, 5)) * px2cm);
    output_table.NumData(i,:) = [sum(neuralData_isDistracted & isNesting), sum((~neuralData_isDistracted) & isNesting)];
    output_table.DataRatio(i) = output_table.NumData(i,1) / sum(output_table.NumData(i,:));
    
end