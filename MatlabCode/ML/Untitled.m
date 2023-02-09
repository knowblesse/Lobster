%% AnalyzeFineDistanceData


load("Apparatus.mat");
px2cm = 0.169;
truncatedTimes_s = 10;
neural_data_rate = 20;
load("D:\Data\Lobster\FineDistanceDataset\#21JAN5-210803-182450_IL\#21JAN5-210803-182450_IL_frameInfo.mat");
load("D:\Data\Lobster\FineDistanceResult_syncFixed\#21JAN5-210803-182450_ILresult_distance.mat");
distracted = readmatrix("D:\Data\Lobster\FineDistanceDataset\#21JAN5-210803-182450_IL\bool_distracted.csv");

midPointTimes = truncatedTimes_s + (1/neural_data_rate)*(0:size(WholeTestResult,1)-1) + 0.5 * (1/neural_data_rate);

%% Construct Frame Number correction
neuralData_isDistracted = logical(distracted(round(interp1(frameTime, frameNumber, midPointTimes))));
isNesting = WholeTestResult(:,2) < 225;


%% Compare Error btw shuffled and predicted
L1_distracted = abs(WholeTestResult(neuralData_isDistracted & isNesting, 3) - WholeTestResult(neuralData_isDistracted & isNesting, 5)) * px2cm;
L1_notDistracted = abs(WholeTestResult(~neuralData_isDistracted & isNesting, 3) - WholeTestResult(~neuralData_isDistracted & isNesting, 5)) * px2cm;
