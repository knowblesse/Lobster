function ts = A2function(filepath,neuronname)
%% Unit basic analysis

load(filepath);

% if ~exist('f','var')
%     [FileName,PathName] = uigetfile;
% 
%     filepath = strcat(PathName,FileName);
% 
%     load(filepath);
% end
% 
% clearvars FileName PathName filepath

%% waveform

su = table2array(SU);
ts = su(:,1); %Timestamps : 언제 firing 했는지를 알려줌.

%% Draw Graph

fig = figure(1);
clf;

subplot(2,1,1);
wav = su(:,4:end);
p = size(wav,2)/4;
x = 1:1:p;

wave1 = wav(:,1:p);
wave2 = wav(:,p+1:2*p);
wave3 = wav(:,2*p+1:3*p);
wave4 = wav(:,3*p+1:4*p);

mwave1 = mean(wave1);
mwave2 = mean(wave2);
mwave3 = mean(wave3);
mwave4 = mean(wave4);

pwave1 = wave1';
pwave2 = wave2';
pwave3 = wave3';
pwave4 = wave4';

plot(x,pwave1,'b');
hold on;
plot(x+p,pwave2,'b');
plot(x+2*p,pwave3,'b');
plot(x+3*p,pwave4,'b');

ylabel('MicroV');

subplot(2,1,2);

plot(x,mwave1,'b','LineWidth',2);
hold on;
plot(x+p,mwave2,'b','LineWidth',2);
plot(x+2*p,mwave3,'b','LineWidth',2);
plot(x+3*p,mwave4,'b','LineWidth',2);

ylabel('MicroV');

saveas(fig,strcat(neuronname,'_1.png'));
end