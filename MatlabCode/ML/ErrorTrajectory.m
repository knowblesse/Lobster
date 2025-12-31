px2cm = 0.169;
distanceRange = [floor(min(WholeTestResult(:,3)) * px2cm), ceil(max(WholeTestResult(:,3)) * px2cm)];

index = discretize(WholeTestResult(:,3) * px2cm,...
    distanceRange(1) : distanceRange(2));

error = (WholeTestResult(:,3) - WholeTestResult(:,5)) * px2cm;

output = zeros(diff(distanceRange),2);

for i = 1 : diff(distanceRange)
    output(i,:) = [mean(WholeTestResult(index == i,5)* px2cm), std(WholeTestResult(index == i,5)* px2cm) ./ sqrt(sum(index == i))];
end

clf;        
shadeplot((distanceRange(1):distanceRange(2)-1)+0.5, output(:,1), 'ShadeValue',output(:,2), 'Color', 'k', 'LineWidth',1);
hold on;
plot((distanceRange(1):distanceRange(2)-1)+0.5, (distanceRange(1):distanceRange(2)-1)+0.5, 'k', 'LineStyle','--')
xlim([10,110]);
ylim([10,110]);
xlabel('True Distance (cm)');
ylabel('Predicted Distance (cm)');



