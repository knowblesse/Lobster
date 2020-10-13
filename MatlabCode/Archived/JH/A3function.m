function A3function(ts, TRON, TROF, IRON, IROF, LICK, LOFF, ATTK, ATOF,neuronname)

%% exp conditions

%  attack at 6s : 80% 
%  attack at 3s : 20%
% 1 block of trials

%% BLock_trial_lize  % cut data by trials

TR = struct;

for j = 1: length(TRON)
    
    TR(j).TRON = TRON(j);
    TR(j).TROF = TROF(j);
                
    k = IRON(TRON(j)<IRON);
    k = k(k<TROF(j));
    TR(j).IRON = k;
    
    k = IROF(TRON(j)<IROF);
    k = k(k<TROF(j));
    TR(j).IROF = k;
   
    k = LICK(TRON(j)<LICK);
    k = k(k<TROF(j));
    TR(j).LICK = k;
     
    k = LOFF(TRON(j)<LOFF);
    k = k(k<TROF(j));
    TR(j).LOFF = k;
    
    k = ATTK(TRON(j)<ATTK);
    k = k(k<TROF(j));
    TR(j).ATTK = k;
    
    k = ATOF(TRON(j)<ATOF);
    k = k(k<TROF(j));
    TR(j).ATOF = k;
    
    k = ts(TRON(j)-10<ts);  % trial+10s before after, get spikes
    k = k(k<TROF(j)+10);
    TR(j).ts = k;
    
end


%% 반복되는 코드 하나로 통일

IRon = cell(1,length(TRON));
IRoff = cell(1,length(TRON));
Lon= cell(1,length(TRON));
Loff = cell(1,length(TRON));
Attk = cell(1,length(TRON));
Atof = cell(1,length(TRON));
stmp = cell(1,length(TRON));
trof = cell(1,length(TRON));

for j = 1 : length(TRON)
    IRon{j} = TR(j).IRON-TRON(j);
    IRoff{j} = TR(j).IROF-TRON(j);
    Lon{j} = TR(j).LICK-TRON(j);
    Loff{j} = TR(j).LOFF-TRON(j);
    Attk{j} = TR(j).ATTK-TRON(j);
    Atof{j} = TR(j).ATOF-TRON(j);
    stmp{j} = TR(j).ts-TRON(j);
    trof{j} = TROF(j)-TRON(j);
end

%% Only Lick trials PRTH (peri response t histogram)
%align by last IR
nodataflag = false; % 데이터에 문제가 있는 경우 true
fig = figure(2);
clf;
    
subplot(4,1,1);
hold on;
subplot(4,1,3);
hold on;
subplot(4,1,4);
hold on;
    
ltr = 0;
s_tmp_stack = [];

for j = 1:length(TRON)
    
    if (~isempty(Lon{j}))
        if numel(IRon{j}) == 0 % 만약 IROn 요소가 하나도 없으면, 
            warning([num2str(j),' 번째 trial 에 IR이 하나도 없음']);
            nodataflag = true;
            clf;
            break;
        end
        
        IR = IRoff{j}(end);
        
        % align
        L_ON = Lon{j}-IR;
        L_OFF = Loff{j}-IR;
        IR_ON = IRon{j}-IR;
        IR_OFF = IRoff{j}-IR;
        A_ttk = Attk{j}-IR;
        A_tof = Atof{j}-IR;
        s_tmp = stmp{j}-IR;
        s_tmp_stack = [s_tmp_stack; s_tmp];
        
        ltr = ltr+1;

        subplot(4,1,1)
        if ~isempty(s_tmp)
            for h = 1:length(s_tmp)
                line([s_tmp(h) s_tmp(h)],[ltr-1 ltr],'Color','k','LineWidth',1)
            end        
        end

        subplot(4,1,3)
        for h = 1:length(IR_ON)
            line([IR_ON(h) IR_OFF(h)],[ltr-1 ltr-1],'Color','g','LineWidth',2)
        end

        subplot(4,1,4)
        for h = 1:length(L_ON)
            line([L_ON(h) L_OFF(h)],[ltr-1 ltr-1],'Color','b','LineWidth',2)
        end
    end  
end

% Label 그리기
subplot(4,1,1);
xlabel('seconds');
xlim([-10 5]);
ylabel('trials');

subplot(4,1,3);
xlabel('seconds');
xlim([-10 5]);
ylabel('trials');

subplot(4,1,4);
xlabel('seconds');
xlim([-10 5]);
ylabel('trials');

if ltr ~=0

    subplot(4,1,1)
    line([0 0],[0 ltr],'Color','g','LineWidth',3)
    hold on

    ylim([0 ltr])

    tt = strcat('all trials aligned by last IR');
    title(tt)

    subplot(4,1,2)

    edges = -10:.1:5; %from -10 befor last lick to 5 after last
    [N, edges] = histcounts(s_tmp_stack, edges);
    Nt = N/ltr;

    bar(Nt)
    set(gca, 'XTick', 0.5:10:length(edges)+0.5)
    set(gca, 'XTickLabel', edges(1:10:end))
    xlim([0.5 length(edges)-.5])

    ylabel('Firing Rate')

end
saveas(fig,neuronname);
end