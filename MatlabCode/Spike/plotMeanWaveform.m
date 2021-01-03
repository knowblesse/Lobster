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
