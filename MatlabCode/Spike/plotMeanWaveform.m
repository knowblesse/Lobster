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
interElectrodeSpace = 10;
figure(2);
clf;
hold on;
for i = 1 : max(floor(size(spikeform,1)/100),1) : size(spikeform,1)
    plot([...
        spikeform(i,1:30),...
        nan(1,interElectrodeSpace),...
        spikeform(i,31:60),...
        nan(1,interElectrodeSpace),...
        spikeform(i,61:90),...
        nan(1,interElectrodeSpace),...
        spikeform(i,91:120),...
        nan(1,interElectrodeSpace),...
        ],'k','LineWidth',0.5);
end
axis off;