%% drawQuiver
% Draw Quiver plot from X,Y data set

%DATA = [X, Y];
STRIDE = 50;
%% Make Mesh
X_range = 50 : STRIDE : 550;
Y_range = 100 : STRIDE : 450;


[X_mesh, Y_mesh] = meshgrid(X_range, Y_range);

U = zeros(numel(Y_range),numel(X_range));
V = zeros(numel(Y_range),numel(X_range));
W = zeros(numel(Y_range),numel(X_range));

%% Find 
X_idx = discretize(DATA(:,1),X_range);
Y_idx = discretize(DATA(:,2),Y_range);

num_point = size(DATA,1);

for i = 1 : num_point - 1
    U(Y_idx(i), X_idx(i)) = U(Y_idx(i), X_idx(i)) + ...
        (DATA(i + 1,1) - DATA(i,1));
    V(Y_idx(i), X_idx(i)) = V(Y_idx(i), X_idx(i)) + ...
        (DATA(i + 1,2) - DATA(i,2));
    W(Y_idx(i), X_idx(i)) = W(Y_idx(i), X_idx(i)) + 1;
end

% for i = 1 : num_point - 1
%     U(Y_idx(i), X_idx(i)) = U(Y_idx(i), X_idx(i)) + ...
%         (DATA(i + 1,1)) - ( X_range(X_idx(i)) + STRIDE/2);
%     V(Y_idx(i), X_idx(i)) = V(Y_idx(i), X_idx(i)) + ...
%         (DATA(i,2)) - ( Y_range(Y_idx(i)) + STRIDE/2);
%     W(Y_idx(i), X_idx(i)) = W(Y_idx(i), X_idx(i)) + 1;
% end

U = U ./ W;
V = V ./ W;

U(isnan(U)) = 0;
V(isnan(V)) = 0;

figure(1);
clf;

hold on;
quiver(X_mesh + (STRIDE/2),Y_mesh + (STRIDE/2),U,V,3, 'LineWidth',3,'Color','y');
plot(DATA(:,1), DATA(:,2),'Color',[0.6,0.6,0.6]);

