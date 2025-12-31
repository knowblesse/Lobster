%% Plot mean spike
spikeform = SU(:,4:end);
m_spikeform = mean(spikeform,1);
std_spikeform = std(spikeform,1);
figure(1);
clf;
hold on;
fill([1:120, 120:-1:1],[m_spikeform + std_spikeform, fliplr(m_spikeform - std_spikeform)],'k','FaceAlpha',0.1,'LineStyle','none');
plot(m_spikeform,'k','LineWidth',2);
axis off;

%% Plot raw spike
% after a discussion with my PI, I changed the design of the plot
spikeform = SU(:,4:end);
interElectrodeSpace = 10;
figure(2);
clf;
hold on;
for i = 3 : max(floor(size(spikeform,1)/100),1) : size(spikeform,1)
    if rem(size(spikeform,2),4) ~= 0
        error('Can not divide spikeform data into 4');
    end
    singleSpikeSize = size(spikeform,2) / 4;
    plot([...
        spikeform(i,1:singleSpikeSize),...
        nan(1,interElectrodeSpace),...
        spikeform(i,singleSpikeSize+1:2*singleSpikeSize),...
        nan(1,interElectrodeSpace),...
        spikeform(i,2*singleSpikeSize+1:3*singleSpikeSize),...
        nan(1,interElectrodeSpace),...
        spikeform(i,3*singleSpikeSize+1:4*singleSpikeSize),...
        nan(1,interElectrodeSpace),...
        ],'k','LineWidth',0.5);
end
axis off;

%% Another version with shade plot
figure(3);
spikeData = reshape(spikeform, [], 30,4);

for electrode = 1 : 4
    subplot(1,4,electrode);
    shadeplot(spikeData(:, :, electrode), 'Color', 'w', 'LineWidth', 1);
    ylim([-8000, 4000]);
    axis off;
end
    


