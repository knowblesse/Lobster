function [ map ] = createGaussianHitmap( xydata, distribution, stride, axis_limit, dodraw )
%xy 데이터로부터 hitmap을 그려냅니다.
%   @param xydata 위치 데이터
%   @param distribution 분산.
%   @param stride 뛰어넘을 데이터. 그리는게 느려지는 경우 사용.
%   @param axis_limit plot 영역 [xmin, xmax, ymin, ymax]
%   @param dodraw true 일 경우에만 heatmap을 그린다.

%% 그려지는 영역
if isempty(axis_limit)
    ax_x = 160 : 580;
    ax_y = 110 : 350;
else
    ax_x = axis_limit(1) : axis_limit(2);
    ax_y = axis_limit(3) : axis_limit(4);
end

[X,Y] = meshgrid(ax_x, ax_y);

%% Multi Variative Gaussian 변수 설정
sigma = ...
    [distribution,0;...
    0,distribution];

%% Hitmap 생성
map = zeros(length(ax_x),length(ax_y));
completeRate = stride / size(xydata,1);
r = 0;
p = gcp;
if p.Connected == true
    dataindex = 1 : stride : size(xydata,1);
    mu_x = xydata(dataindex,1);
    mu_y = xydata(dataindex,2);
    X = X(:);
    Y = Y(:);
    parfor i = 1 : numel(dataindex)
        mu = [mu_x(i),mu_y(i)];
        F = mvnpdf([X,Y],mu,sigma);
        F = reshape(F,length(ax_y), length(ax_x));
        map = map + F';
    end
else
    for i = 1 : stride : size(xydata,1)
        mu = xydata(i,:);
        F = mvnpdf([X(:),Y(:)],mu,sigma);
        F = reshape(F,length(ax_y), length(ax_x));
        map = map + F';
        r = r + completeRate;
        t = r * 100;
        fprintf([num2str(t),'\n']);
    end
end
    



map = map ./ (size(xydata,1)/stride);

if dodraw == true
    hitmap = surf(ax_x,ax_y,map', 'EdgeColor','none');
end

end

