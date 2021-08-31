load('RTP/RTP_NYC_2010_2019.mat')
load('DAP/DAP_NYC_2010_2019.mat')
load('centorid.mat')
load('cluster_value_function.mat')
Ts = 1/12; % time step
DD = 365; % select days to look back
la = 1; % look ahead hours
lambda = reshape(RTP(:,(end-DD):end),numel(RTP(:,(end-DD):end)),1); 
lambda_DA = reshape(DAP(:,(end-DD):end),numel(DAP(:,(end-DD):end)),1);
bias = lambda -lambda_DA;
T = numel(lambda)-24/Ts; % number of time steps

%%
Pr = .5; % normalized power rating wrt energy rating
P = Pr*Ts; % actual power rating taking time step size into account
eta = .9; % efficiency
c = 10; % marginal discharge cost - degradation
ed = .001; % SoC sample granularity
ef = .5; % final SoC target level, use 0 if none
Ne = floor(1/ed)+1; % number of SOC samples
e0 = .5;

%%
Xtest = zeros(T,la/Ts+1);
Xtest(:,1) = lambda_DA(24/Ts+1:end);

for tp = 1:la/Ts
    Xtest(:,tp+1) = bias(24/Ts+1-tp:end-tp);
end

 [~,idx_test] = pdist2(C,Xtest,'cityblock','Smallest',1);

%% perform the actual arbitrage
eS = zeros(T,1); % generate the SoC series
pS = eS; % generate the power series
vEnd = zeros(Ne,1);  % generate value function samples
vEnd(1:floor(ef*100)) = 1e2;

e = e0; % initial SoC
vv=zeros(Ne,T);
for t = 1:T-1 % start from the first day and move forwards
   vv(:,t) = vc(:,idx_test(t+1)); % read the SoC value for this day
   [e, p] =  Arb_Value(lambda(t+24/Ts), vv(:,t), e, P, 1, eta, c, size(vc,1));
   eS(t) = e; % record SoC
   pS(t) = p; % record Power
end
vv(:,end) = vEnd;
[e, p] =  Arb_Value(lambda(end), vv(:,t), e, P, 1, eta, c, size(vc,1));
eS(end) = e; % record SoC
pS(end) = p; % record Power


ProfitOut = sum(pS.*lambda(24/Ts+1:end)) - sum(c*pS(pS>0));
Revenue = sum(pS.*lambda(24/Ts+1:end));
fprintf('Profit=%e, revenue=%e',ProfitOut, Revenue)