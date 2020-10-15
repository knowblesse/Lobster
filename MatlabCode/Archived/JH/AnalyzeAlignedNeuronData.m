%% depricated


%% aligned mat 파일 여러개를 한꺼번에 읽어서 PCA를 돌리는 스크립트
% Imlazy_createmat.m 이나 Imlazy_trial_by_trial.m 에서 생성하는 *aligned.mat 데이터가
% 필요.
% 각각의 aligned.mat 은 하나의 neuron에 대응하는 데이터임.
% 이 코드를 실행하면 여러 aligned.mat 파일을 선택하도록 하고, 이 데이터로 PCA 분석을 진행함.

%% Constants
%varnames = {'TRON_mat','IRON_mat','LICK_mat','LOFF_mat','IROF_mat','ATTK_mat','TROF_mat'};
varnames = {'LOFF_mat','IROF_mat','ATTK_mat'};
zeroline = 40; % A3에서 edges를 기준으로 0인 지점의 index 값. A3 코드가 바뀌면 바꿔줘야 함!!!
xedges = -4:0.1:4; 

N = 4; % 사용하는 Component의 수
KernelSize = 10; % moving average의 크기

%% PCA 돌리고 그 결과값(loading)에 따라서 재정렬)
Screensize = get(groot, 'Screensize'); % 좌에 정렬된 그래프, 우에 eigenvector를 표시하기 위한 부분.

for events = 1 : numel(varnames)
    % PCA 돌리기. 이 부분은 논문의 코드와 정확히 일치함.
    X = eval(varnames{events}); % 각 조건별 데이터 불러오기
    X = movmean(X,KernelSize,2); % X축으로 이동하며(열로 이동하며) moving average)
    [u, s, v] = svd(X',0);
    U = u(:,1:N); % eigenvenctors
    l = diag(s).^2/(numNeuron-1); % eigenvalues
    Z = X*U; % scores = loading
    clearvars u s v
    
    % Component figure 그리기.
    gui2 = figure('name',[varnames{events}, ' : Components']);
    set(gui2, 'Position', [1,1,Screensize(3)/2,Screensize(4)]);
    movegui(gui2,'east');
    for comp = 1 : N
        subplot(N,1,comp);
        plot(U(:,comp));
    end
    subplot(N,1,1);
    title([varnames{events}, ' : Components']);
    
    % loading 값에 따라서 정렬한 Neuron 데이터 그리기.
    gui1 = figure('name',varnames{events});
    set(gui1, 'Position', [1,1,Screensize(3)/2,Screensize(4)]);
    movegui(gui1,'west');
    % 총 N개의 component를 사용. 각 component의 loading 값을 기준으로 정렬함.
    % 데이터에는 따로 손을 안대고 정렬하는 방식만 바꾼 것.
    title(varnames{events});
    for comp = 1 : N
        subplot(1,N,comp);
        [~,sortindex] = sort(Z(:,comp));
        imagesc(X(sortindex,:));
        colormap jet;
        hold on;
        line([zeroline, zeroline],[0, numNeuron],'Color','r');
    end
    subplot(1,N,1);
    title(varnames{events});
end
        

