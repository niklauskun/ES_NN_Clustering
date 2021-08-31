load('RTP/RTP_NYC_2010_2019.mat')
load('DAP/DAP_NYC_2010_2019.mat')
Ts = 1/12;
lastDay = datenum(2019,12,31);
start = datenum(2017,1,1); % start point of training data
stop = datenum(2018,12,31); % end point of training data
la = 1; % look ahead hours
cluster = 10;
tlambda = reshape(RTP(:,(end-lastDay+start-1):(end-lastDay+stop)),numel(RTP(:,(end-lastDay+start-1):(end-lastDay+stop))),1);
tlambda_DA = reshape(DAP(:,(end-lastDay+start-1):(end-lastDay+stop)),numel(DAP(:,(end-lastDay+start-1):(end-lastDay+stop))),1);
% tlambda = reshape(RTP(:,(end-1):end),numel(RTP(:,(end-1):end)),1); 
% tlambda_DA = reshape(DAP(:,(end-1):end),numel(DAP(:,(end-1):end)),1);
tbias = tlambda -tlambda_DA;
T = (stop-start+1)*24/Ts; % number of time steps
% T = 288; % number of time steps

%% Storage parameters
Pr = .5; % normalized power rating wrt energy rating
P = Pr*Ts; % actual power rating taking time step size into account
eta = .9; % efficiency
c = 10; % marginal discharge cost - degradation
ed = .001; % SoC sample granularity
ef = .5; % final SoC target level, use 0 if none
Ne = floor(1/ed)+1; % number of SOC samples
e0 = .5;

vEnd = zeros(Ne,1);  % generate value function samples

vEnd(1:floor(ef*100)) = 1e2;
% vEnd(1:floor(ef*(Ne-1))) = 1e2; % use 100 as the penalty for final discharge level

%%
tic
v = zeros(Ne, T+1); % initialize the value function series
% v(1,1) is the marginal value of 0% SoC at the beginning of day 1
% V(Ne, T) is the maringal value of 100% SoC at the beginning of the last operating day
v(:,end) = vEnd; % update final value function

% process index
es = (0:ed:1)';
Ne = numel(es);
% calculate soc after charge vC = (v_t(e+P*eta))
eC = es + P*eta; 
% round to the nearest sample 
iC = ceil(eC/ed)+1;
iC(iC > (Ne+1)) = Ne + 2;
iC(iC < 2) = 1;
% calculate soc after discharge vC = (v_t(e-P/eta))
eD = es - P/eta; 
% round to the nearest sample 
iD = floor(eD/ed)+1;
iD(iD > (Ne+1)) = Ne + 2;
iD(iD < 2) = 1;

for t = T:-1:1 % start from the last day and move backwards
    vi = v(:,t+1); % input value function from tomorrow
    vo = CalcValueNoUnc(tlambda(t+24/Ts), c, P, eta, vi, ed, iC, iD);
    v(:,t) = vo; % record the result 
end


%%
% preprocess features
X = zeros(T,la/Ts+1);
X(:,1) = tlambda_DA(24/Ts+1:end); % include day-ahead price prediction

for tp = 1:la/Ts
    X(:,tp+1) = tbias(24/Ts+1-tp:end-tp); % include real-time pirce as fetures (12 timepoints before current timepoint)
end

opts = statset('Display','final','MaxIter',1000000);
[idx,C] = kmeans(X,cluster,'Distance','cityblock','Options',opts);

% calculate mean value function of each clusters
vc = zeros(Ne,cluster);
for i = 1:cluster
    cidx = find(idx==i);
    vc(:,i)=mean(v(:,cidx),2);
end

save('centorid.mat','C')
save('cluster_value_function.mat','vc')
