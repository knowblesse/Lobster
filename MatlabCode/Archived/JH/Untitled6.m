
TARGET_EVENT = 'IROF';

numA = eval(strcat('size(A.',TARGET_EVENT,',1);'));
numE = eval(strcat('size(E.',TARGET_EVENT,',1);'));
%numS = eval(strcat('size(S.',TARGET_EVENT,',1);'));
%EVENT = eval(strcat('[A.',TARGET_EVENT,';E.',TARGET_EVENT,';S.',TARGET_EVENT,'];'));
EVENT = eval(strcat('[A.',TARGET_EVENT,';E.',TARGET_EVENT,'];'));

coeff = pca(EVENT);

for i = 1 : 80
    PC{i} = EVENT * coeff(:,i);
end
clear i

% A, E pair 거리 구하기
pca_A = [PC{1}(1:numA),PC{2}(1:numA),PC{3}(1:numA)];
pca_E = [PC{1}(numA+1:numA+numE),PC{2}(numA+1:numA+numE),PC{3}(numA+1:numA+numE)];

distance = sum((pca_A - pca_E).^2,2) .^ 0.5;
[~,indices] = sort(distance);
colors = jet(numA); % Avoid Escape 쌍의 갯수로 colorrange를 설정.

figure(2);
clf

scatter3(PC{1}(1:numA),PC{2}(1:numA),PC{3}(1:numA),'r','*');
hold on;
scatter3(PC{1}(numA+1:numA+numE),PC{2}(numA+1:numA+numE),PC{3}(numA+1:numA+numE),'g','*');
hold on;
scatter3(PC{1}(numA+numE+1:end),PC{2}(numA+numE+1:end),PC{3}(numA+numE+1:end),'b','*');


for i = 1 : numA
%     scatter3(PC{1}(i),PC{2}(i),PC{3}(i),'*','MarkerFaceColor',colors(i,:));
%     hold on;
%     scatter3(PC{1}(numA+i),PC{2}(numA+i),PC{3}(numA+i),'*','MarkerFaceColor',colors(i,:));
    scatter3(PC{1}(i),PC{2}(i),PC{3}(i),'r','*');
    hold on;
    scatter3(PC{1}(numA+i),PC{2}(numA+i),PC{3}(numA+i),'g','*');
end

xlabel('PC1');
ylabel('PC2');
zlabel('PC3');