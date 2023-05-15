%% LabelSpatialMaps
fig = figure();


addpath('..');
global maps c data apparatus
load('SpatialMaps.mat');
load('Apparatus.mat');

maps = spatialMaps;
c = 1;
data = zeros(632, 1); % 0: not responsive, 1: one peak, 2: multiple peak, 3:Ezone-aisle
refresh();


fig.KeyReleaseFcn = @keyEvent;

function refresh()
    global maps c apparatus
    h = imshow(squeeze(maps(c, :, :, :)));
    set(h, 'AlphaData', apparatus.mask);
    caxis([-0.2,0.2]);
    title(num2str(c));
end

function keyEvent(src, event)
    global data c
    if event.Character == '0'
        data(c) = 0;
    elseif event.Character == '1'
        data(c) = 1;
    elseif event.Character == '2'
        data(c) = 2;
    elseif event.Character == '3'
        data(c) = 3;
    else
        return;
    end
    c = c + 1;
    refresh();
end

    