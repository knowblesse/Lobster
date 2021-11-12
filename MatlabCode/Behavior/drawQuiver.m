%% drawQuiver
% Draw Quiver plot from buttered data set

csv_path = 'D:\Data\Lobster\Lobster_Recording-200319-161008\21JAN5\#21JAN5-210813-182242_IL\Lobster_Recording-210330-101307_21JAN5-210813-182242_Vid1_buttered.csv';
data = readmatrix(csv_path); % frame num | row | col | degree (South position is East position is 45 degree)

data_row = data(:,2);
data_col = data(:,3);
load('EmptyApparatus.mat');

%% Make Mesh
DIVIDER = 25;
X_range = round(linspace(1, size(image,2), DIVIDER));
Y_range = round(linspace(1, size(image,1), DIVIDER));

[X_mesh, Y_mesh] = meshgrid(X_range, Y_range);

U = zeros(numel(Y_range),numel(X_range)); % arrow X element
V = zeros(numel(Y_range),numel(X_range)); % arrow Y element
W = zeros(numel(Y_range),numel(X_range)); % Normalize Factor

%% Find 
X_idx = discretize(data(:,3),X_range);
Y_idx = discretize(data(:,2),Y_range);

num_point = size(data,1);

for i = 1 : num_point
    U(Y_idx(i), X_idx(i)) = U(Y_idx(i), X_idx(i)) + cosd(data(i,4));
    V(Y_idx(i), X_idx(i)) = V(Y_idx(i), X_idx(i)) + sind(data(i,4));
    W(Y_idx(i), X_idx(i)) = W(Y_idx(i), X_idx(i)) + 1;
end

U = U ./ W;
V = V ./ W;

U(isnan(U)) = 0;
V(isnan(V)) = 0;

figure(2);
clf;
imshow(image);

hold on;
quiver(X_mesh,Y_mesh,U,V,0.7, 'LineWidth',2,'Color','y','AlignVertexCenters',false);
