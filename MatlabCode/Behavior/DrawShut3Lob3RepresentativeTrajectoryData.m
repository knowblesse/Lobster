path_shut3 = "D:\Data_fib\Lobster_Training-190725-161613\21JAN5-210224-111236\Lobster_Training-190725-161613_21JAN5-210224-111236_Vid1_buttered.csv";
path_lob1 = "D:\Data_fib\Lobster_Training-190725-161613\21JAN5-210225-115452\Lobster_Training-190725-161613_21JAN5-210225-115452_Vid1_buttered.csv";

%% Load Data
data_shut3 = readmatrix(path_shut3);
data_lob1 = readmatrix(path_lob1);
load("EmptyApparatus.mat");

%% Draw Figure
figure(1);

subplot(1,2,1);
imagesc(apparatus);
hold on;
plot(data_shut3(:,3), data_shut3(:,2), 'Color', [1, 1, 1, 0.3]);
axis off;

subplot(1,2,2);
imagesc(apparatus);
hold on;
plot(data_lob1(:,3), data_lob1(:,2), 'Color', [1, 1, 1, 0.3]);
axis off;

%%
data = data_lob1;
%clf;
imagesc(apparatus);
hold on;
plot(data(:,3), data(:,2), 'Color', [1, 1, 1, 0.3]);

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

for i = 1 : size(endIndex,1)
    plot(...
        data(startIndex(i) : endIndex(i), 3),...
        data(startIndex(i) : endIndex(i), 2),...
        'Color', 'r');
end

