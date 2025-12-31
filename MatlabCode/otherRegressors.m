px2cm = 0.169;

reg1 = mean(abs(WholeTestResult(:, 3) - WholeTestResult(:,4))) * px2cm
reg2 = mean(abs(WholeTestResult(:, 3) - WholeTestResult(:,5))) * px2cm
reg3 = mean(abs(WholeTestResult(:, 3) - WholeTestResult(:,6))) * px2cm
reg4 = mean(abs(WholeTestResult(:, 3) - WholeTestResult(:,7))) * px2cm
