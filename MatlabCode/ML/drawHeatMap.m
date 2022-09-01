load('EventClassificationResult.mat');

PLdata = result(contains(tankNames, "PL"));
ILdata = result(contains(tankNames, "IL"));

cm_shuffled = zeros(numel(PLdata), 3, 3);
cm_real = zeros(numel(PLdata), 3, 3);
for i = 1 : numel(PLdata)
    cm = double(PLdata{i}.confusion_matrix);
    cm_shuffled(i, :, :) = squeeze(cm(1, :, :)) ./ repmat(sum(squeeze(cm(1, :, :)), 2), 1, 3);
    cm_real(i, :, :) = squeeze(cm(2, :, :)) ./ repmat(sum(squeeze(cm(2, :, :)), 2), 1, 3);
end

fig = figure('Name', 'PL');

subplot(2,2,1);
heatmap(squeeze(mean(cm_shuffled, 1)));
caxis([0, 1]);

subplot(2,2,2);
heatmap(squeeze(mean(cm_real, 1)));
caxis([0, 1]);



cm_shuffled = zeros(numel(ILdata), 3, 3);
cm_real = zeros(numel(ILdata), 3, 3);
for i = 1 : numel(ILdata)
    cm = double(ILdata{i}.confusion_matrix);
    cm_shuffled(i, :, :) = squeeze(cm(1, :, :)) ./ repmat(sum(squeeze(cm(1, :, :)), 2), 1, 3);
    cm_real(i, :, :) = squeeze(cm(2, :, :)) ./ repmat(sum(squeeze(cm(2, :, :)), 2), 1, 3);
end

subplot(2,2,3);
heatmap(squeeze(mean(cm_shuffled, 1)));
caxis([0, 1]);

subplot(2,2,4);
heatmap(squeeze(mean(cm_real, 1)));
caxis([0, 1]);