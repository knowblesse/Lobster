%% FromHE2HW_PETH
% Draw HE to HW PETH graph
% Since the duration between the HE and the HW varies across trials, 
% interpolation method is included.
% Beware.
% This methods is more like the method for generating the ML classifiers' 
% dataset, not the z score for the PETHs.

addpath('../FeatureImportance/');
ClassifyUnits;

%% Change this to draw other group
%indices = find(Unit.Group_HE == 1 & Unit.Group_HW == 1);
indices = find(Unit.Group_HW == 2);

%% Sufficiently large matrix for storing long spike data
outputMatrix = zeros(numel(indices), 5000); % -1000ms from HE | 3000ms | 1000ms from HW

%% Build Gaussian Kernel (Same parameters as Event Classifier)
KERNEL_SIZE = 1000;
KERNEL_STD = 100;

kernel = gausswin(ceil(KERNEL_SIZE/2)*2-1, (KERNEL_SIZE - 1) / (2 * KERNEL_STD)); % kernel size is changed into an odd number for symmetrical kernel application. see Matlab gausswin docs for the second parameter.

%% Go through all units
for i = 1 : numel(indices)
    index = indices(i);

    spikes = Unit.RawSpikeData{index};
    
    [timepoint,numTrial] = getTimepointFromParsedData(Unit.BehavData{index});
    
    %% Serialize timestamp data(Sampling frequency = 1000Hz)
    spk = round(spikes);
    % use max(10s from last spike, 10s from the last TROF) as the length of the serial data
    serial_data = zeros(round(max(spk(end) + (10*1000), (timepoint.TROF(end)/1000 + 10)*1000)),1);
    serial_data(spk,1) = 1;
    
    serial_data_kerneled =  conv(serial_data,kernel,'same');
    
    % Get mean and std of serialized signal for normalization
    serial_data_mean = mean(serial_data_kerneled);
    serial_data_std = std(serial_data_kerneled);
    
    % For every trial, interpolate HE ~ HW data into 10000 length
    tempMatrix = zeros(numTrial,10000);
    for trial = 1 : numTrial    
        startTime = round(timepoint.first_LICK(trial) - 1000); % 1 s behind
        endTime = round(timepoint.valid_IROF(trial) + 1000);
        convolvedData = conv(serial_data(startTime:endTime),kernel,'same');
        tempMatrix(trial, :) = interp1(1:numel(convolvedData), convolvedData, linspace(1, numel(convolvedData), 10000));
    end
    
    meanActivity = (mean(tempMatrix,1) - serial_data_mean) ./ serial_data_std;

    outputMatrix(i, :) = meanActivity;
end
