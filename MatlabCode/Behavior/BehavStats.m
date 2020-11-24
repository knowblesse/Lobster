%% Plot Behav Stats
% Plotting behavior related data

%% Load Behavior event data
% for concatenating multiple files
NUM_FILES = 1;
ParsedData_ = {};
behaviorResult_ = [];
numIRClusters_ = [];
numLickClusters_ = [];
for f = 1 : NUM_FILES
    [ParsedData, Trials, IRs, Licks, Attacks ] = BehavDataParser();
    behaviorResult = analyticValueExtractor(ParsedData, false, false);
    ParsedData_ = [ParsedData_;ParsedData];
    behaviorResult_ = [behaviorResult_; behaviorResult];
end

ParsedData = ParsedData_;
behaviorResult = behaviorResult_;
numIRClusters = numIRClusters_;
numLickClusters = numLickClusters_;

clearvars ParsedData_ behaviorResult_ numIRClusters_ numLickClusters_;
    
%% Data Array Format
% +-----+-----+-------+-------+--------+----------------+
% |  1  |  2  |   3   |   4   |   5    |       6        |
% +-----+-----+-------+-------+--------+----------------+
% | fIR | lIR | fLick | lLick | Attack | BehaviorResult |
% +-----+-----+-------+-------+--------+----------------+
% BehaviorResult : A : 0 | E : 1 | G : 2 | M : 3
numTrial = size(ParsedData,1);
InputArray = zeros(numTrial, 6);

for t = 1 : numTrial
    switch(behaviorResult(t))
        case 'A'
            InputArray(t,6) = 0;
        case 'E'
            InputArray(t,6) = 1;
        case 'G'
            InputArray(t,6) = 2;
        case 'M'
            InputArray(t,6) = 3;
    end        
    if or(InputArray(t,6) == 0, InputArray(t,6) == 1) % Avoid나 Escape 인 경우에만.
        InputArray(t,1) = ParsedData{t,2}(1); % first IR
        InputArray(t,2) = ParsedData{t,2}(end); % last IR
        InputArray(t,3) = ParsedData{t,3}(1); % first Lick
        InputArray(t,4) = ParsedData{t,3}(end); % last Lick
        InputArray(t,5) = ParsedData{t,4}(1); % fAttack
    end
end
clear t;

%% Data for Stats
% +-----+-----------+-------+-------------+-------------+--------------+------------+---------------+-----------------+----------------+
% |  1  |     2     |   3   |      4      |      5      |      6       |     7      |       8       |        9        |       10       |
% +-----+-----------+-------+-------------+-------------+--------------+------------+---------------+-----------------+----------------+
% | fIR | length IR | fLick | length Lick | fLick - fIR | Attk - lLick | Attk - lIR | numIRClusters | numLickClusters | BehaviorResult |
% +-----+-----------+-------+-------------+-------------+--------------+------------+---------------+-----------------+----------------+

StatArray = zeros(numTrial,10);
StatArray(:,1) = InputArray(:,1);
StatArray(:,2) = InputArray(:,2) - InputArray(:,1);
StatArray(:,3) = InputArray(:,3);
StatArray(:,4) = InputArray(:,4) - InputArray(:,3);
StatArray(:,5) = InputArray(:,3) - InputArray(:,1);
StatArray(:,6) = InputArray(:,5) - InputArray(:,4);
StatArray(:,7) = InputArray(:,5) - InputArray(:,2);
% 8 and 9 is deprecated. 
StatArray(:,10) = InputArray(:,2) - InputArray(:,3);
StatArray(:,11) = InputArray(:,6);
%% Draw Figures
Monitors = get(groot, 'MonitorPositions');

%% First IR
fig1 = figure('Name','First IR');
set(fig1,'Position',[1,1,Monitors(1,3),Monitors(1,4)]);
draw6graph(1, [0,20,-inf,inf], 0:1:20,numTrial, StatArray);

%% Length IR
fig2 = figure('Name','Length between the first and the last IR');
set(fig2,'Position',[1,1,Monitors(1,3),Monitors(1,4)]);
draw6graph(2, [0,15,-inf,inf], 0:0.5:15, numTrial, StatArray);

%% First Lick
fig3 = figure('Name','First Lick');
set(fig3,'Position',[1,1,Monitors(1,3),Monitors(1,4)]);
draw6graph(3, [0,30,-inf,inf], 0:1:30, numTrial, StatArray);

%% Length Lick
fig4 = figure('Name','Length between the first and the last Lick');
set(fig4,'Position',[1,1,Monitors(1,3),Monitors(1,4)]);
draw6graph(4, [0,7,-inf,inf], 0:0.1:7, numTrial, StatArray);

%% fLick - fIR
fig5 = figure('Name','Length between the first Lick and the first IR');
set(fig5,'Position',[1,1,Monitors(1,3),Monitors(1,4)]);
draw6graph(5, [0,10,-inf,inf], 0:0.1:10, numTrial, StatArray);

%% Attack - lLick
fig6 = figure('Name','Length between the Attack and the last Lick');
set(fig6,'Position',[1,1,Monitors(1,3),Monitors(1,4)]);
draw6graph(6, [-0.5,6,-inf,inf],-0.5:0.1:6, numTrial, StatArray);

%% Attack - lIR
fig7 = figure('Name','Length between the Attack and the last IR');
set(fig7,'Position',[1,1,Monitors(1,3),Monitors(1,4)]);
draw6graph(7, [-0.5,6,-inf,inf], -0.5:0.1:6, numTrial, StatArray);

%% IR Cluster
fig8 = figure('Name','IR clusters');
set(fig8,'Position',[1,1,Monitors(1,3),Monitors(1,4)]);
draw6graph(8, [0,5,-inf,inf], 0:5, numTrial, StatArray);

%% Lick Cluster
fig9 = figure('Name','Lick clusters');
set(fig9,'Position',[1,1,Monitors(1,3),Monitors(1,4)]);
draw6graph(9, [0,5,-inf,inf], 0:6, numTrial, StatArray);

%% lIR - fLick
fig10 = figure('Name','Length between the first Lick and the last IR');
set(fig10,'Position',[1,1,Monitors(1,3),Monitors(1,4)]);
draw6graph(10, [0,7,-inf,inf], 0:0.1:7, numTrial, StatArray);

%% Draw Composition Data
fig_C = figure('Name','Trial Composition');
set(fig_C,'Position',[1,1,1200,300]);
movegui(fig_C,'center');

% sequential behav pattern
for i = numTrial : -1 : 1
    switch(StatArray(i,11))
        case 0
            barh(2,i,'FaceColor',[0.507,0.789,0.984]); % Avoid 인 경우 파란색
            hold on;
        case 1
            barh(2,i,'FaceColor',[0.965,0.527,0.602]); % Escape 인 경우 빨간색
            hold on;
        case 2
            barh(2,i,'FaceColor',[0.8,0.8,0.8]); % Give Up 인 경우 회색
            hold on;
        case 3
            barh(2,i,'k'); % 1Min Out 인 경우 흰색
            hold on;
    end
end

% behav pattern composition
tempdat = [...
    sum(StatArray(:,11) == 0),... % Avoid
    sum(StatArray(:,11) == 1),... % Escape
    sum(StatArray(:,11) == 2),... % Give Up
    sum(StatArray(:,11) == 3),... % 1Min Out
    ];
compdat = cumsum(tempdat);

barh(1,compdat(4),'k');
barh(1,compdat(3),'FaceColor',[0.8,0.8,0.8])
barh(1,compdat(2),'FaceColor',[0.965,0.527,0.602]);
barh(1,compdat(1),'FaceColor',[0.507,0.789,0.984]);

text(tempdat(1)/2, 1,'A','FontSize',12,'FontWeight','bold','Color','w');
text(tempdat(2)/2 + compdat(1), 1,'E','FontSize',12,'FontWeight','bold','Color','w');
text(tempdat(3)/2 + compdat(2), 1,'G','FontSize',12,'FontWeight','bold','Color','k');
text(tempdat(4)/2 + compdat(3), 1,'M','FontSize',12,'FontWeight','bold','Color','w');

yticks([1,2]);
yticklabels({'Composition','Behavior Type'});

title('Trial Composition');
xlabel('trial');

% 사용자 편의를 위해서 버튼 만들기
figs = {fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10};

for i = 1 : 10
    figs{i}.Visible = 'off';
end

ui_leftstart = 180;
ui_interval = 90;

pushbutton_1 = uicontrol('Style','pushbutton','String','First IR',...
    'Position',[ui_leftstart,60,80,30],...
    'Callback',@figvis,...
    'UserData',{1,figs} ...
    );
pushbutton_2 = uicontrol('Style','pushbutton','String','IR Length',...
    'Position',[ui_leftstart+ui_interval*1,60,80,30],...
    'Callback',@figvis,...
    'UserData',{2,figs} ...
    );
pushbutton_3 = uicontrol('Style','pushbutton','String','First Lick',...
    'Position',[ui_leftstart+ui_interval*2,60,80,30],...
    'Callback',@figvis,...
    'UserData',{3,figs} ...
    );
pushbutton_4 = uicontrol('Style','pushbutton','String','Lick Length',...
    'Position',[ui_leftstart+ui_interval*3,60,80,30],...
    'Callback',@figvis,...
    'UserData',{4,figs} ...
    );
pushbutton_5 = uicontrol('Style','pushbutton','String','IR to Lick',...
    'Position',[ui_leftstart+ui_interval*4,60,80,30],...
    'Callback',@figvis,...
    'UserData',{5,figs} ...
    );
pushbutton_6 = uicontrol('Style','pushbutton','String','lLick to Attk',...
    'Position',[ui_leftstart+ui_interval*5,60,80,30],...
    'Callback',@figvis,...
    'UserData',{6,figs} ...
    );
pushbutton_7 = uicontrol('Style','pushbutton','String','lIR to Attk',...
    'Position',[ui_leftstart+ui_interval*6,60,80,30],...
    'Callback',@figvis,...
    'UserData',{7,figs} ...
    );
pushbutton_8 = uicontrol('Style','pushbutton','String','IR Cluster',...
    'Position',[ui_leftstart+ui_interval*7,60,80,30],...
    'Callback',@figvis,...
    'UserData',{8,figs} ...
    );
pushbutton_9 = uicontrol('Style','pushbutton','String','Lick Cluster',...
    'Position',[ui_leftstart+ui_interval*8,60,80,30],...
    'Callback',@figvis,...
    'UserData',{9,figs} ...
    );
pushbutton_10 = uicontrol('Style','pushbutton','String','fLick to lIR',...
    'Position',[ui_leftstart+ui_interval*9,60,80,30],...
    'Callback',@figvis,...
    'UserData',{10,figs} ...
    );

%% Define Repetitive Function
function draw6graph(type, axis_, bin_, numTrial, StatArray)
    subplot(2,3,1);
    xbar = 1:numTrial;
    barh(flipud(xbar(StatArray(:,11) == 0)),flipud(StatArray(StatArray(:,11) == 0,type)),'FaceColor',[0.507,0.789,0.984]);
    hold on;
    barh(flipud(xbar(StatArray(:,11) == 1)),flipud(StatArray(StatArray(:,11) == 1,type)),'FaceColor',[0.965,0.527,0.602]);
    hold off;
    axis(axis_);
    title('All Trials');
    xlabel('sec');
    ylabel('trials');
    tick = 0:10:(ceil(numTrial/10))*10;
    yticks(tick);
    yticklabels(fliplr(tick)+1);
    legend({'avoid','escape'});

    subplot(2,3,4);
    histogram(StatArray(:,type),bin_);
    axis(axis_);
    xlabel('sec');
    
    % Avoid trial graph
    subplot(2,3,2);
    barh(flipud(StatArray(StatArray(:,11) == 0,type)),'FaceColor',[0.507,0.789,0.984]);
    line([mean(StatArray(StatArray(:,11) == 0,type)),mean(StatArray(StatArray(:,11) == 0,type))],...
        get(gca,'YLim'),'Color','k');
    temp = get(gca,'YLim');
    text(mean(StatArray(StatArray(:,11) == 0,type)),temp(2)/2,num2str(mean(StatArray(StatArray(:,11) == 0,type))),'FontSize',15);
    axis(axis_);
    title('Avoid');
    xlabel('sec');
    ylabel('trials');
    yticks([]);

    % Escape trial graph
    subplot(2,3,3);
    barh(flipud(StatArray(StatArray(:,11) == 1,type)),'FaceColor',[0.965,0.527,0.602]);
        line([mean(StatArray(StatArray(:,11) == 1,type)),mean(StatArray(StatArray(:,11) == 1,type))],...
        get(gca,'YLim'),'Color','k');
    temp = get(gca,'YLim');
    text(mean(StatArray(StatArray(:,11) == 1,type)),temp(2)/2,num2str(mean(StatArray(StatArray(:,11) == 1,type))),'FontSize',15);
    axis(axis_);
    title('Escape');
    xlabel('sec');
    ylabel('trials');
    yticks([]);

    % Avoid Histo
    subplot(2,3,5);
    histogram(StatArray(StatArray(:,11) == 0,type),bin_);
    axis(axis_);
    xlabel('sec');

    % Escape Histo
    subplot(2,3,6);
    histogram(StatArray(StatArray(:,11) == 1,type),bin_);
    axis(axis_);
    xlabel('sec');          
end
function figvis(hObject, ~, ~)
figs = hObject.UserData{2};
for i = 1 : 10
    figs{i}.Visible = 'off';
end
    figs{hObject.UserData{1}}.Visible = 'on';
end
