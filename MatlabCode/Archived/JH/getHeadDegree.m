function [ head_degree ] = getHeadDegree( x1, y1, x2, y2 )
%이전좌표와 이후 좌표를 기반으로 움직인 방향의 각도를 구합니다.
head_radian = atan(...
    (y2 - y1)/...
    (x2 - x1));
% vector(1,0) 을 0 도로 기준으로 90도까지는 직관적인 표시가 되지만 270~360도 사이는 마이너스 값으로 표시가 되고
% 90~270까지는 값이 -90~90 사이로 나오기에 다른 값들과 겹침.
% 이에 (90,270)에 해당하는 값들은 따로 처리를 해줄 필요가 있음.
if x1 <= x2 % [-90,90] 범위 안에 드는 녀석들.
    head_degree = rad2deg(head_radian);
else
    head_degree = rad2deg(head_radian) + 180;
end

