%% script to learn PINN for SIM approximation and compare with GSPT-derived ones
clear
clc

%rng default; % for reproducibility

%% set dimensions before calling the generating function
inSz = 4;   % number of input vars z = (u,v) 
outSz = 4;  % number of output vars (x, y) 
Mfast = 2;  % number of assumed fast vars: x
noHL = 1;   % number of hidden layers
hlSz = 20;   % and neurons on them
firstTrain = true;
RPnoRuns = 100;  % number of random runs

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DATA PREPARATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% set the parameters for full model
KM1 = 23;
k2 = 42/60;
k1b = 4*k2;
k1f = 5*k2/KM1;
KM2 = 25;
k4 = 25/60;
k3b = 4*k4;
k3f = 5*k4/KM2;
e0 = 50;

load CompInh_TrainData dataTrain;
load CompInh_TestData allData;

pars = [k1f k1b k2 k3f k3b k4 e0];
z2sol = [2; 4]; % corresponding to c1, c2
fImpNR = true;  
[QSSAc1c2_SIM, PEA13c1c2_SIM, CSP11c1c2_SIM, CSP21c1c2_SIM] = CompInh_knownSIMs(allData(2:5,:),pars,z2sol,fImpNR);

%% many runs to get averages
CPUrecs = zeros(RPnoRuns,1);
trainMSE = zeros(RPnoRuns,2);
CV_MSE = zeros(RPnoRuns,2);
trainLinf = zeros(RPnoRuns,2);
CV_Linf = zeros(RPnoRuns,2);
trainL2 = zeros(RPnoRuns,2);
CV_L2 = zeros(RPnoRuns,2);
learnedAll = zeros(RPnoRuns,inSz^2+Mfast*(hlSz*(inSz-Mfast+2)+1));
for i = 1:RPnoRuns
    %% Form Data Sets
    XData = dataTrain(end-outSz+1:end,:);              % solution of ODE system
    % random selection of test set
    idx = randperm(size(XData,2),floor(size(XData,2)*0.2));
    Xtrain_CV = XData(:,idx);
    Xtrain = XData;
    Xtrain(:,idx) = [];

    %% precompute RHS to save time from F(z)=dzdt
    Fz = INHode(0., Xtrain,k1f,k1b,k2,k3f,k3b,k4,e0);

    tic
    %% set architecture and initialize NNs
    [learnables, netDim] = prepareNetwork(inSz,noHL,hlSz,outSz,Mfast,Xtrain);
    
    %% Train PIML
    if firstTrain
        %% Initial train to configure Transformation
        options=optimoptions(@lsqnonlin,'Display','none','MaxIter',1500,'TolFun',1e-5,...
            'Algorithm','levenberg-marquardt','MaxFunEvals',1000000,'TolX',1e-6,...
            'UseParallel',false,'SpecifyObjectiveGradient',true);%,'FinDiffRelStep',1e-6,'CheckGradients',true,'FiniteDifferenceType','central');
        [learned1,resnorm1,RES1,fExit1,Out1] = lsqnonlin(@(curLearn) funPIloss(curLearn,netDim,Xtrain,Fz,2),learnables,[],[],options);
        %% Subsequently train to minimize errors in Transformation, using weights.
        options.MaxIterations = 500;
%         options.TolFun = 1e-8;
%         options.TolX = 1e-6;
        [learned2,resnorm2,RES2,fExit2,Out2] = lsqnonlin(@(curLearn) funPIloss(curLearn,netDim,Xtrain,Fz,1),learned1,[],[],options);
        %% Round the Transformation and optimize the NN
        TransLearned1 = learned2(1,1:inSz^2);      % reshape is per columns
        TransLearned = round(TransLearned1,2);  % round it to 1e-3;
        if (sum(TransLearned) >= inSz-1e-5) || (sum(TransLearned) <= inSz+1e-5)
            learnables = learned2(1,inSz^2+1:end);
            options.MaxIterations = 500;
            % options.TolFun = 1e-10;
            % options.TolX = 1e-8;
            [learned,resnorm,RES,fExit,Out] = lsqnonlin(@(curLearn) funPIloss(curLearn,netDim,Xtrain,Fz,3,TransLearned),...
                learnables,[],[],options);
        else
            learned = learned2(1,inSz^2+1:end);
            TransLearned
        end
        learned = [TransLearned learned];
    end
    CPUend = toc;
    CPUrecs(i,1) = CPUend;

    %% collect learnables    
    learnedAll(i,:) = learned;
    %
    %% SLFNN train error and CV error
    XzTr = Xtrans(Xtrain,learned,netDim);
    YzTr = Ytrans(Xtrain,learned,netDim);
    trainSIM = forwardNN3(YzTr,learned,netDim);
    for j = 1:Mfast
        trainMSE(i,j) = mse(trainSIM(j,:),XzTr(j,:));             %% X(z)-h(Y(z))
        trainLinf(i,j) = norm(trainSIM(j,:)-XzTr(j,:),Inf);
        trainL2(i,j) = norm(trainSIM(j,:)-XzTr(j,:),2);
    end
    XzCV = Xtrans(Xtrain_CV,learned,netDim);
    YzCV = Ytrans(Xtrain_CV,learned,netDim);
    CV_SIM = forwardNN3(YzCV,learned,netDim);
    for j = 1:Mfast
        CV_MSE(i,j) = mse(CV_SIM(j,:),XzCV(j,:));   
        CV_Linf(i,j) = norm(CV_SIM(j,:)-XzCV(j,:),Inf);
        CV_L2(i,j) = norm(CV_SIM(j,:)-XzCV(j,:),2);
    end
end

%% Confidence intervals of training
ts = tinv([0.025 0.05 0.1 0.9 0.95 0.975],RPnoRuns-1);   % T-Score
muCPU = mean(CPUrecs); stdCPU = std(CPUrecs);
muT2 = mean(trainL2); stdT2 = std(trainL2);
muTinf = mean(trainLinf); stdTinf = std(trainLinf);
muTmse = mean(trainMSE); stdTmse = std(trainMSE);
muV2 = mean(CV_L2); stdV2 = std(CV_L2);
muVinf = mean(CV_Linf); stdVinf = std(CV_Linf);
muVmse = mean(CV_MSE); stdVmse = std(CV_MSE);
semCPU = stdCPU/sqrt(RPnoRuns);                          % Standard Errors
semT2 = stdT2/sqrt(RPnoRuns); semTinf = stdTinf/sqrt(RPnoRuns); semTmse = stdTmse/sqrt(RPnoRuns);
semV2 = stdV2/sqrt(RPnoRuns); semVinf = stdVinf/sqrt(RPnoRuns); semVmse = stdVmse/sqrt(RPnoRuns);
ciCPU = muCPU + ts*semCPU;                               % Confidence Intervals
ciT2 = muT2' + semT2'.*ts; ciTinf = muTinf' + semTinf'.*ts; ciTmse = muTmse' + semTmse'.*ts;
ciV2 = muV2' + semV2'.*ts; ciVinf = muVinf' + semVinf'.*ts; ciVmse = muVmse' + semVmse'.*ts;

%% Metrics on training
fprintf('-------Training metrics--------\n')
fprintf('CPU times:           mean             std             5-95 CI   \n');
fprintf('               %e      %e      %e   %e \n',muCPU,stdCPU,ciCPU(2),ciCPU(5));
fprintf('Errors on the first variable! \n')
fprintf('Errors (L2):         mean             std             5-95 CI   \n');
fprintf('Train Set:     %e      %e      %e   %e \n',muT2(1,1),stdT2(1,1),ciT2(1,2),ciT2(1,5));
fprintf('CV Set:        %e      %e      %e   %e \n',muV2(1,1),stdV2(1,1),ciV2(1,2),ciV2(1,5));
fprintf('Errors (Linf):       mean             std             5-95 CI   \n');
fprintf('Train Set:     %e      %e      %e   %e \n',muTinf(1,1),stdTinf(1,1),ciTinf(1,2),ciTinf(1,5));
fprintf('CV Set:        %e      %e      %e   %e \n',muVinf(1,1),stdVinf(1,1),ciVinf(1,2),ciVinf(1,5));
fprintf('Errors (MSE):        mean             std             5-95 CI   \n');
fprintf('Train Set:     %e      %e      %e   %e \n',muTmse(1,1),stdTmse(1,1),ciTmse(1,2),ciTmse(1,5));
fprintf('CV Set:        %e      %e      %e   %e \n',muVmse(1,1),stdVmse(1,1),ciVmse(1,2),ciVmse(1,5));
fprintf('Errors on the second variable! \n')
fprintf('Errors (L2):         mean             std             5-95 CI   \n');
fprintf('Train Set:     %e      %e      %e   %e \n',muT2(1,2),stdT2(1,2),ciT2(2,2),ciT2(2,5));
fprintf('CV Set:        %e      %e      %e   %e \n',muV2(1,2),stdV2(1,2),ciV2(2,2),ciV2(2,5));
fprintf('Errors (Linf):       mean             std             5-95 CI   \n');
fprintf('Train Set:     %e      %e      %e   %e \n',muTinf(1,2),stdTinf(1,2),ciTinf(2,2),ciTinf(2,5));
fprintf('CV Set:        %e      %e      %e   %e \n',muVinf(1,2),stdVinf(1,2),ciVinf(2,2),ciVinf(2,5));
fprintf('Errors (MSE):        mean             std             5-95 CI   \n');
fprintf('Train Set:     %e      %e      %e   %e \n',muTmse(1,2),stdTmse(1,2),ciTmse(2,2),ciTmse(2,5));
fprintf('CV Set:        %e      %e      %e   %e \n',muVmse(1,2),stdVmse(1,2),ciVmse(2,2),ciVmse(2,5));

Tmetrics = array2table([muCPU stdCPU ciCPU; muT2(1,1) stdT2(1,1) ciT2(1,:); muTinf(1,1) stdTinf(1,1) ciTinf(1,:);...
    muTmse(1,1) stdTmse(1,1) ciTmse(1,:); muV2(1,1) stdV2(1,1) ciV2(1,:); muVinf(1,1) stdVinf(1,1) ciVinf(1,:); ...
    muVmse(1,1) stdVmse(1,1) ciVmse(1,:); muT2(1,2) stdT2(1,2) ciT2(2,:); muTinf(1,2) stdTinf(1,2) ciTinf(2,:);...
    muTmse(1,2) stdTmse(1,2) ciTmse(2,:); muV2(1,2) stdV2(1,2) ciV2(2,:); muVinf(1,2) stdVinf(1,2) ciVinf(2,:); ... 
    muVmse(1,2) stdVmse(1,2) ciVmse(2,:)],...
    "VariableNames",{'mean','std','CI 2.5','CI 5.0','CI 10.0','CI 90.0','CI 95.0','CI 97.5'},...
    "RowNames",{'CPU','TrainL2 1st','TrainLinf 1st','TrainLmse 1st','ValidL2 1st','ValidLinf 1st','ValidLmse 1st',...
    'TrainL2 2nd','TrainLinf 2nd','TrainLmse 2nd','ValidL2 2nd','ValidLinf 2nd','ValidLmse 2nd'});
save CompInh_trainMetric Tmetrics;

%% Test Set errors 
%% PIML
norm2trainMSE = zeros(RPnoRuns,1);
for i=1:RPnoRuns
    norm2trainMSE(i,1) = norm(trainMSE(i,:),2);
end
[~,idx] = min(norm2trainMSE);
bestLearned = learnedAll(idx,:);
Xz = Xtrans(allData(2:5,:),bestLearned,netDim);
Yz = Ytrans(allData(2:5,:),bestLearned,netDim);
PIML_SIM = forwardNN3(Yz,bestLearned,netDim);
PIML_MSE = [mse(PIML_SIM(1,:),Xz(1,:)); mse(PIML_SIM(2,:),Xz(2,:))];             %% X(z)-h(Y(z))
PIML_Linf = [norm(PIML_SIM(1,:)-Xz(1,:),Inf); norm(PIML_SIM(2,:)-Xz(2,:),Inf)];
PIML_L2 = [norm(PIML_SIM(1,:)-Xz(1,:),2); norm(PIML_SIM(2,:)-Xz(2,:),2)];
%% GSPT expressions, HERE all expressions are solved for z2 (explicit or implicit)
z2All = [allData(1 + z2sol(1,1),:); allData(1 + z2sol(2,1),:)];
QSSAc1c2_MSE = [mse(QSSAc1c2_SIM(1,:),z2All(1,:)); mse(QSSAc1c2_SIM(2,:),z2All(2,:))];                                      %% z2_QSSAL - z2
if ~fImpNR           %% implicit errors 
    PEA13c1c2_MSE = [mse(PEA13c1c2_SIM(1,:),zeros(size(z2All(1,:)))); mse(PEA13c1c2_SIM(2,:),zeros(size(z2All(2,:))));];    %% h_CSPL21(z1,z2,z3,z4)-0
    CSP11c1c2_MSE = [mse(CSP11c1c2_SIM(1,:),zeros(size(z2All(1,:)))); mse(CSP11c1c2_SIM(2,:),zeros(size(z2All(2,:))));];    %% h_CSPL21(z1,z2,z3,z4)-0
    CSP21c1c2_MSE = [mse(CSP21c1c2_SIM(1,:),zeros(size(z2All(1,:)))); mse(CSP21c1c2_SIM(2,:),zeros(size(z2All(2,:))));];    %% h_CSPL21(z1,z2,z3,z4)-0
else                %% explicit with NR
    PEA13c1c2_MSE = [mse(PEA13c1c2_SIM(1,:),z2All(1,:)); mse(PEA13c1c2_SIM(2,:),z2All(2,:))]; 
    CSP11c1c2_MSE = [mse(CSP11c1c2_SIM(1,:),z2All(1,:)); mse(CSP11c1c2_SIM(2,:),z2All(2,:))]; 
    CSP21c1c2_MSE = [mse(CSP21c1c2_SIM(1,:),z2All(1,:)); mse(CSP21c1c2_SIM(2,:),z2All(2,:))]; 
end
QSSAc1c2_Linf = [norm(QSSAc1c2_SIM(1,:)-z2All(1,:),Inf); norm(QSSAc1c2_SIM(2,:)-z2All(2,:),Inf)]; 
if ~fImpNR           %% implicit errors 
    PEA13c1c2_Linf = [norm(PEA13c1c2_SIM(1,:),Inf); norm(PEA13c1c2_SIM(2,:),Inf);];    %% h_CSPL21(z1,z2,z3,z4)-0
    CSP11c1c2_Linf = [norm(CSP11c1c2_SIM(1,:),Inf); norm(CSP11c1c2_SIM(2,:),Inf);];    %% h_CSPL21(z1,z2,z3,z4)-0
    CSP21c1c2_Linf = [norm(CSP21c1c2_SIM(1,:),Inf); norm(CSP21c1c2_SIM(2,:),Inf);];    %% h_CSPL21(z1,z2,z3,z4)-0
else                %% explicit with NR
    PEA13c1c2_Linf = [norm(PEA13c1c2_SIM(1,:)-z2All(1,:),Inf); norm(PEA13c1c2_SIM(2,:)-z2All(2,:),Inf)]; 
    CSP11c1c2_Linf = [norm(CSP11c1c2_SIM(1,:)-z2All(1,:),Inf); norm(CSP11c1c2_SIM(2,:)-z2All(2,:),Inf)]; 
    CSP21c1c2_Linf = [norm(CSP21c1c2_SIM(1,:)-z2All(1,:),Inf); norm(CSP21c1c2_SIM(2,:)-z2All(2,:),Inf)]; 
end
QSSAc1c2_L2 = [norm(QSSAc1c2_SIM(1,:)-z2All(1,:),2); norm(QSSAc1c2_SIM(2,:)-z2All(2,:),2)]; 
if ~fImpNR           %% implicit errors 
    PEA13c1c2_L2 = [norm(PEA13c1c2_SIM(1,:),2); norm(PEA13c1c2_SIM(2,:),2);];    %% h_CSPL21(z1,z2,z3,z4)-0
    CSP11c1c2_L2 = [norm(CSP11c1c2_SIM(1,:),2); norm(CSP11c1c2_SIM(2,:),2);];    %% h_CSPL21(z1,z2,z3,z4)-0
    CSP21c1c2_L2 = [norm(CSP21c1c2_SIM(1,:),2); norm(CSP21c1c2_SIM(2,:),2);];    %% h_CSPL21(z1,z2,z3,z4)-0
else                %% explicit with NR
    PEA13c1c2_L2 = [norm(PEA13c1c2_SIM(1,:)-z2All(1,:),2); norm(PEA13c1c2_SIM(2,:)-z2All(2,:),2)]; 
    CSP11c1c2_L2 = [norm(CSP11c1c2_SIM(1,:)-z2All(1,:),2); norm(CSP11c1c2_SIM(2,:)-z2All(2,:),2)]; 
    CSP21c1c2_L2 = [norm(CSP21c1c2_SIM(1,:)-z2All(1,:),2); norm(CSP21c1c2_SIM(2,:)-z2All(2,:),2)]; 
end
fprintf('Test Set errors for z1: on data of SIM \n')
fprintf('L2  :   PIML(e)     QSSAc1c2(e)    PEA13c1c2(i)     CSP11c1c2(i)    CSP21c1c2(i)\n');
fprintf('      %10.3e    %10.3e     %10.3e       %10.3e       %10.3e   \n',PIML_L2(1),QSSAc1c2_L2(1),PEA13c1c2_L2(1),CSP11c1c2_L2(1),CSP21c1c2_L2(1));
fprintf('Linf:    \n');
fprintf('      %10.3e    %10.3e     %10.3e       %10.3e       %10.3e   \n',PIML_Linf(1),QSSAc1c2_Linf(1),PEA13c1c2_Linf(1),CSP11c1c2_Linf(1),CSP21c1c2_Linf(1));
fprintf('MSE :    \n');
fprintf('      %10.3e    %10.3e     %10.3e       %10.3e       %10.3e   \n',PIML_MSE(1),QSSAc1c2_MSE(1),PEA13c1c2_MSE(1),CSP11c1c2_MSE(1),CSP21c1c2_MSE(1));
fprintf('Test Set errors for z2: on data of SIM \n')
fprintf('L2  :   PIML(e)     QSSAc1c2(e)    PEA13c1c2(i)     CSP11c1c2(i)    CSP21c1c2(i)\n');
fprintf('      %10.3e    %10.3e     %10.3e       %10.3e       %10.3e   \n',PIML_L2(2),QSSAc1c2_L2(2),PEA13c1c2_L2(2),CSP11c1c2_L2(2),CSP21c1c2_L2(2));
fprintf('Linf:    \n');
fprintf('      %10.3e    %10.3e     %10.3e       %10.3e       %10.3e   \n',PIML_Linf(2),QSSAc1c2_Linf(2),PEA13c1c2_Linf(2),CSP11c1c2_Linf(2),CSP21c1c2_Linf(2));
fprintf('MSE :    \n');
fprintf('      %10.3e    %10.3e     %10.3e       %10.3e       %10.3e   \n',PIML_MSE(2),QSSAc1c2_MSE(2),PEA13c1c2_MSE(2),CSP11c1c2_MSE(2),CSP21c1c2_MSE(2));
if ~fImpNR
    fprintf('Implicit forms are not solved for the fast variable numerically!\n');
else
    fprintf('Implicit forms are solved for the fast variable with Newton numerically!\n');
end


return



%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Physics-informed loss function 
%    
%  minimization of SIMEq                1) X(z)-H(Y(z)) = NN1(z) - NN3(NN2(z))
%                  InvEq(grad of SIM)   2) [I, -dHdY]*[dXdz; dYdz]*dzdt = [I -dNN3_dy]*[dNN1_dz; dNN2_dz]*RHS(z) 
%                  Transformation related loss  3) see text
function [TotMin, Jac] = funPIloss(curLearn,netDim,UVin,Fz,fDim,TransLearned)
    %% Stop learning the transformation from now on
    if fDim == 3
        if nargin<=5; error('Supply the transformation');end
        curLearn = [TransLearned curLearn];
    end
    %% get Tranformations: z->x and z->y and their derivatives
    [Xz, dXdz] = Xtrans(UVin,curLearn,netDim);
    [Yz, dYdz] = Ytrans(UVin,curLearn,netDim);
    %% get SIMEq: y->x
    [NN3Xy, dNdy, ddNNdy] = forwardNN3(Yz,curLearn,netDim);

    %% Form argmins (1): X(z)-H(Y(z))
    FminSIM = Xz-NN3Xy;   % size MxC
   
    %
    %% Form argmins (2): [I, -dHdY]*[dXdz; dYdz]*F(z)
    AugTr = [dXdz(:,:,1); dYdz(:,:,1)];
    FminIE = zeros(netDim(9),size(UVin,2));   % Mfast x C
    for i=1:size(UVin,2)
        FminIE(:,i) = [eye(netDim(9),netDim(9)) -dNdy(:,:,i)]*AugTr*Fz(:,i);
    end

    %% Transformation conditions: x = X(z) and y=Y(z) are linear combinations  
    TransSumR = sum((AugTr),2)-1;                       % sum of each row
    TransSumCF = sum((AugTr(1:netDim(3),:)),1);         % sum of Columns for fast vars
    TransSumCS = sum((AugTr(netDim(3)+1:end,:)),1);     % sum of Columns for slow vars
    TransSumC = TransSumCF+TransSumCS-1;                % sum of columns is 1
    TransMultCS1 = TransSumCF.*AugTr(netDim(3)+1:end,:);  % totally separated fast and slow
    TransMultCS2 = TransSumCS.*AugTr(1:netDim(3),:);

    %% collection of optimization function with pinning conditions
    if fDim==1
        weight = 100;
        TotMin = [reshape(FminSIM',[],1) ; reshape(FminIE',[],1) ; ...    %% column per m, WATCH OUT
            reshape(weight*TransSumC,[],1) ; reshape(weight*TransSumR,[],1) ; ...
            reshape(weight*TransMultCS1',[],1); reshape(weight*TransMultCS2',[],1)];
    elseif fDim==2
        weight = 1;
        TotMin = [reshape(FminSIM',[],1) ; reshape(FminIE',[],1) ; ...    %% column per m, WATCH OUT
            reshape(weight*TransSumC,[],1) ; reshape(weight*TransSumR,[],1) ; ...
            reshape(weight*TransMultCS1',[],1); reshape(weight*TransMultCS2',[],1)]; 
    elseif fDim==3
        TotMin = [reshape(FminSIM',[],1) ; reshape(FminIE',[],1)];
        if (sum(abs(TransSumC))~=0) || (sum(abs(TransSumR))~=0) || (sum(abs(TransMultCS1),'all')~=0) || (sum(abs(TransMultCS2),'all')~=0)
            error('Smth went wrong with the roundoff of the Transformation');
        end
    end

    if nargout > 1
        sz1 = size(FminSIM,2);
        Mf = netDim(3);
        Ntot = netDim(1);
        hlSz = netDim(2);
        if (fDim == 1) || (fDim == 2)
            Jac_shift = 0;
        elseif fDim == 3
            Jac_shift = -Ntot^2;
        end
        Jac = zeros(size(TotMin,1),size(curLearn,2)+Jac_shift);

        %% FminSIM ders
        if (fDim == 1) || (fDim == 2)
            % Xz trans ders
            for m = 1:Mf
                Jac((m-1)*sz1+1:m*sz1,(m-1)*Ntot+1:m*Ntot) = UVin';
            end
            % Yz trans ders
            for m = 1:Mf
                for nmm = 1:Ntot-Mf
                    Jac((m-1)*sz1+1:m*sz1,Mf*Ntot +(nmm-1)*Ntot+1:Mf*Ntot +nmm*Ntot) = - squeeze(dNdy(m,nmm,:)).*UVin';
                end
            end
        end
        %% Get all NN derivatives (used also next)
        % NN pars ders
        [dNNdWo, ddNNdWody, dNNdBo, ddNNdBody, dNNdW, ddNNdWdy, dNNdB, ddNNdBdy] = gradsNN(Yz, curLearn, netDim);
        totParNo = (Ntot-Mf+2)*hlSz+1; % number of learnables per m
        for m = 1:Mf
            % Win ders
            for nmm = 1:Ntot-Mf
                Jac((m-1)*sz1+1:m*sz1,Jac_shift+Ntot^2+(m-1)*totParNo+(nmm-1)*hlSz+1:Jac_shift+Ntot^2+(m-1)*totParNo+nmm*hlSz) = ...
                    -squeeze(dNNdW(m,m,nmm,:,:))';
            end
            % bin ders
            Jac((m-1)*sz1+1:m*sz1,Jac_shift+Ntot^2+(Ntot-Mf)*hlSz+(m-1)*totParNo+1:Jac_shift+Ntot^2+(Ntot-Mf+1)*hlSz+(m-1)*totParNo) = ...
                -squeeze(dNNdB(m,m,:,:))';
            % Wout ders
            Jac((m-1)*sz1+1:m*sz1,Jac_shift+Ntot^2+(Ntot-Mf+1)*hlSz+(m-1)*totParNo+1:Jac_shift+Ntot^2+(Ntot-Mf+2)*hlSz+(m-1)*totParNo) = ...
                -squeeze(dNNdWo(m,m,:,:))';
            % bout ders
            Jac((m-1)*sz1+1:m*sz1,Jac_shift+Ntot^2+(Ntot-Mf+2)*hlSz+(m-1)*totParNo+1) = -squeeze(dNNdBo(m,m,:,:));
        end

        %% FminIE ders
        sz2 = size(FminIE,2);
        if (fDim == 1) || (fDim == 2)
            % Xz trans ders
            for m = 1:Mf
                Jac(Mf*sz1+(m-1)*sz2+1:Mf*sz1+m*sz2,(m-1)*Ntot+1:m*Ntot) = Fz';            
            end
            % Yz trans ders
            for m = 1:Mf % fix m
                for nmm = 1:Ntot-Mf  % fix d
                    dNdyd = squeeze(dNdy(m,nmm,:))';
                    for j = 1:Ntot   % fix j
                        ddNm_dyh_dDdj = squeeze(ddNNdy(m,nmm,:,:)).*UVin(j,:);
                        Jac(Mf*sz1+(m-1)*sz2+1:Mf*sz1+m*sz2,Mf*Ntot+(nmm-1)*Ntot+j) = - (dNdyd.*Fz(j,:))' - (sum(AugTr(Ntot-Mf+1:Ntot,:)'*ddNm_dyh_dDdj.*Fz,1))';
                    end
                end      
            end
        end
        % NN pars ders
        temp2 = zeros(Ntot-Mf,totParNo,sz2);
        for m = 1:Mf % counts for r as well cause only diagonal in this case
            for h = 1:Ntot-Mf
                temp2(:,(h-1)*hlSz+1:h*hlSz,:) = squeeze(ddNNdWdy(m,:,m,h,:,:));
            end
            temp2(:,(Ntot-Mf)*hlSz+1:(Ntot-Mf+1)*hlSz,:) = squeeze(ddNNdBdy(m,:,m,:,:));
            temp2(:,(Ntot-Mf+1)*hlSz+1:(Ntot-Mf+2)*hlSz,:) = squeeze(ddNNdWody(m,:,m,:,:));
            temp2(:,totParNo,:) = squeeze(ddNNdBody(m,:,m,:,:));
            for j = 1:Ntot
                sum1 = zeros(sz2,totParNo);
                for nmm = 1:Ntot-Mf
                    sum1 = sum1 + (squeeze(temp2(nmm,:,:))*AugTr(Mf+nmm,j))';
                end
                % I only have non-zeros when m = r
                Jac(Mf*sz1+(m-1)*sz2+1:Mf*sz1+m*sz2,Jac_shift+Ntot^2+(m-1)*totParNo+1:Jac_shift+Ntot^2+m*totParNo) = ...
                    Jac(Mf*sz1+(m-1)*sz2+1:Mf*sz1+m*sz2,Jac_shift+Ntot^2+(m-1)*totParNo+1:Jac_shift+Ntot^2+m*totParNo) - sum1.*Fz(j,:)';
            end
        end

        %% Transformation ders
        if (fDim == 1) || (fDim == 2)
            %% Transformation ders
            % sum of cols - 1
            for m = 1:Mf
                Jac(Mf*(sz1+sz2)+1:Mf*(sz1+sz2)+Ntot,(m-1)*Ntot+1:m*Ntot) = eye(Ntot);
            end
            for nmm = 1:Ntot-Mf
                Jac(Mf*(sz1+sz2)+1:Mf*(sz1+sz2)+Ntot,Mf*Ntot +(nmm-1)*Ntot+1:Mf*Ntot +nmm*Ntot) = eye(Ntot);
            end
            % sum of rows -1 
            for m = 1:Mf
                Jac(Mf*(sz1+sz2)+Ntot+m,(m-1)*Ntot+1:m*Ntot) = ones(1,Ntot); 
            end
            for nmm = 1:Ntot-Mf
                Jac(Mf*(sz1+sz2)+Ntot+Mf+nmm,Mf*Ntot +(nmm-1)*Ntot+1:Mf*Ntot +nmm*Ntot) = ones(1,Ntot);
            end
            % sum of fast * every slow, per j = 1:Ntot i.e. per blocks of Ntot (verticaly and horizontaly)
            for nmm = 1:Ntot-Mf
                Jac(Mf*(sz1+sz2)+2*Ntot+(nmm-1)*Ntot+1:Mf*(sz1+sz2)+2*Ntot+nmm*Ntot,1:Ntot*Mf) = repmat(diag(AugTr(Mf+nmm,:)),1,Mf);
                Jac(Mf*(sz1+sz2)+2*Ntot+(nmm-1)*Ntot+1:Mf*(sz1+sz2)+2*Ntot+nmm*Ntot,Ntot*Mf+(nmm-1)*Ntot+1:Ntot*Mf+nmm*Ntot) = ...
                    diag(TransSumCF);
            end
            % sum of slow * every fast, per j=1:nTot, i.e., per blocs of Ntot (ver and hor)
            for m = 1:Mf
                Jac(Mf*(sz1+sz2)+2*Ntot+(Ntot-Mf)*Ntot+(m-1)*Ntot+1:Mf*(sz1+sz2)+2*Ntot+(Ntot-Mf)*Ntot+m*Ntot,...
                    (m-1)*Ntot+1:m*Ntot) = diag(TransSumCS); 
                Jac(Mf*(sz1+sz2)+2*Ntot+(Ntot-Mf)*Ntot+(m-1)*Ntot+1:Mf*(sz1+sz2)+2*Ntot+(Ntot-Mf)*Ntot+m*Ntot,...
                    Ntot*Mf+1:Ntot^2) = repmat(diag(AugTr(m,:)),1,Ntot-Mf);
            end 
            Jac(Mf*(sz1+sz2)+1:Mf*(sz1+sz2)+2*Ntot+Ntot^2,:) = weight*Jac(Mf*(sz1+sz2)+1:Mf*(sz1+sz2)+2*Ntot+Ntot^2,:); 
        end
    end

end

% function to calculate the gradients of the NN w.r.t. parameters
%
function [dNNdWo, ddNNdWody, dNNdBo, ddNNdBody, dNNdW, ddNNdWdy, dNNdB, ddNNdBdy] = gradsNN(Yin, learnables, netDim)
    [Win, bin, Wout, bout] = unravelLearn(learnables,netDim,3);
    hlSz = size(Win,2);
    inSz = size(Win,3);
    Mf = size(Win,1);
    Ns = size(Yin,2);
    %
    phiL = zeros(Mf,hlSz,Ns);
    dphiL = zeros(Mf,hlSz,Ns);
    for m = 1:Mf
        [phiL(m,:,:), dphiL(m,:,:), ~] = activationFun(squeeze(Win(m,:,:))*Yin+bin(m,:)');
    end
    %
    dNNdWo = zeros(Mf,Mf,hlSz,Ns);                                 %%%%%%%% M x M x L x Ns
    for m = 1:Mf
        dNNdWo(m,m,:,:) = phiL(m,:,:);   % non-diagonal are zeros
    end
    ddNNdWody = zeros(Mf,inSz,Mf,hlSz,Ns);                         %%%%%%%% M x (N-M) x M x L x Ns 
    for m = 1:Mf
        WinPm = squeeze(Win(m,:,:));
        for nmm = 1:inSz
            ddNNdWody(m,nmm,m,:,:) = WinPm(:,nmm).*squeeze(dphiL(m,:,:));  % non-diagonal again are zeros
        end
    end
    %
    dNNdB = zeros(Mf,Mf,hlSz,Ns);                                  %%%%%%%% M x M x L x Ns
    for m = 1:Mf
        dNNdB(m,m,:,:) = squeeze(Wout(m,:,:)).*squeeze(dphiL(m,:,:));    % non-diagonal again are zeros
    end
    ddNNdBdy = zeros(Mf,inSz,Mf,hlSz,Ns);                          %%%%%%%% M x (N-M) x M x L x Ns
    for m = 1:Mf
        WinPm = squeeze(Win(m,:,:));
        for nmm = 1:inSz
            ddNNdBdy(m,nmm,m,:,:) = squeeze(dNNdB(m,m,:,:)).*WinPm(:,nmm).*(1-2*squeeze(phiL(m,:,:))); % non-diagonal again are zeros
        end
    end
    %
    dNNdBo = zeros(Mf,Mf,1,Ns);                                    %%%%%%%% M x M x 1 x Ns
    for m = 1:Mf
        dNNdBo(m,m,1,:) = 1;  % non-diagonal again are zeros
    end
    ddNNdBody = zeros(Mf,inSz,Mf,1,Ns);                            %%%%%%%% M x (N-M) x M x 1 x Ns
    %
    dNNdW = zeros(Mf,Mf,inSz,hlSz,Ns);                             %%%%%%%% M x M x (N-M) x L x Ns
    for m = 1:Mf
        for nmm = 1:inSz
            dNNdW(m,m,nmm,:,:) = squeeze(dNNdB(m,m,:,:)).*Yin(nmm,:); % non-diagonal are zeros
        end
    end
    ddNNdWdy = zeros(Mf,inSz,Mf,inSz,hlSz,Ns);                     %%%%%%%% M x (N-M) x M x (N-M) x L x Ns
    for m = 1:Mf
        % non-diagonals are zeros in terms of M, but not in N-M (2 indices h and d)
        for nmm = 1:inSz % this is d
            for h = 1:inSz
                ddNNdWdy(m,nmm,m,h,:,:) = squeeze(ddNNdBdy(m,nmm,m,:,:)).*Yin(h,:);
                if nmm == h
                    ddNNdWdy(m,nmm,m,h,:,:) = squeeze(ddNNdBdy(m,nmm,m,:,:)).*Yin(h,:) + ...
                        squeeze(dNNdB(m,m,:,:)); % diagonal element
                end
            end
        end
    end
  
end


%% function for calculating X(z) output: from z->x=X(z) and derivatives x_z
function [XofZ, dXdz] = Xtrans(UVin,learnables,netDim)
    C1_a = unravelLearn(learnables,netDim, 1);
    %% forward the input
    XofZ = C1_a*UVin;
    if nargout>1
        dXdz = repmat(C1_a,[1,1,size(UVin,2)]);
    end
end

%% function for calculating Y(z) output: from z->y=Y(z) and derivatives y_z
function [YofZ, dYdz] = Ytrans(UVin,learnables,netDim)
    D2_c = unravelLearn(learnables,netDim, 2);
    %% forward the input
    YofZ = D2_c*UVin;
    if nargout>1
        dYdz = repmat(D2_c,[1,1,size(UVin,2)]);
    end
end

%% function for calculating NN3 output: from y->x=H(y) and derivatives x_y
function [NN3_out, dNN3_dy, ddNN3_dy] = forwardNN3(Yin,learnables,netDim)
    [Win, bin, Wout, bout] = unravelLearn(learnables,netDim,3);
    
    %% forward the input
    NN3_out = zeros(netDim(9),size(Yin,2)); % M x n
    if nargout>1
        dNN3_dy = zeros(netDim(9),netDim(7),size(Yin,2)); %% dhdy is Mx(N-M) for each input point
        ddNN3_dy = zeros(netDim(9),netDim(7),netDim(7),size(Yin,2));  % d(dhdy)/dy is Mx(N-M)x(N-M) for each input point
    end
    for i=1:netDim(9)   %% counts on m=1..M
        if size(Win,3)==1
            temp1 = Win(i,:)';          % format must be Lx(N-M) 
        else
            temp1 = squeeze(Win(i,:,:));
        end
        [phi, dphi, ddphi] = activationFun(temp1*Yin+bin(i,:)');
        temp2 = squeeze(Wout(i,1,:)); %% squeezes also the second dimension
        NN3_out(i,:) = temp2'*phi+bout(i,1);
        if nargout>1
            for j = 1:netDim(7) %% counts on d=1..N-M
                dNN3_dy(i,j,:) = temp2'*(temp1(:,j).*dphi);
                if nargout>2                
                    for k = 1:netDim(7) %% counts on h=1..N-M
                        ddNN3_dy(i,j,k,:) = temp2'*(temp1(:,j).*temp1(:,k).*ddphi);
                    end
                end
            end         
        end
    end
end

%% function for NN architecture and initialization
function [learnables, netDim] = prepareNetwork(inSz,noHL,hlSz,outSz,Mfast,UVin)
    maxIn = max(UVin,[],'all');
    minIn = min(UVin,[],'all');
    rangeIn = 1/(maxIn-minIn);

    if inSz~=outSz; error('Transformation matrices cannot accept different input-output dimensions'); end
    %%%%%%%%%%%%%%
    %
    %% Trans: x = X(z) = C z,  y = Y(z) = D z
    %
    % are supposed to be linear; i.e. x = a z1 + b z2, y = c z1 + d z2
    %%%%%%%%%%%%%%

    %% C matrix: input state vector z, output fast variables x = Az where A:MxN
    C1_inSz = inSz;                        % N
    C1_outSz = Mfast;                      % M
    C1_a = ones(C1_outSz,C1_inSz);
    C1_a = C1_a/C1_inSz;
    
    %% D matrix: input state vector z, output slow variables x
    D2_inSz = inSz;                        % N
    D2_outSz = outSz-Mfast;                % N-M
    D2_c = ones(D2_outSz,D2_inSz);
    D2_c = D2_c/D2_inSz;

    %%%%%%%%%%%%%%
    %
    %% NN SIMEq: NN3 x = H(y)
    %
    %%%%%%%%%%%%%%

    %% NN3: input slow variables y, output fast variables x
    NNdc_inSz = inSz-Mfast;                 % N-M
    NNdc_outSz = Mfast;                     % M               
    NNdc_hlSz = hlSz;                       % number of neurons per hidden layer
    XavierUniR1 = sqrt(6/(NNdc_inSz+NNdc_hlSz))*rangeIn;
    XavierUniR2 = sqrt(6/(NNdc_hlSz+NNdc_outSz))*rangeIn;
    %
    %% 3D arrays with first dimension measuring the (m) output
    NNdc_Win = -XavierUniR1+2*XavierUniR1*rand(NNdc_outSz,NNdc_hlSz,NNdc_inSz);   % input weights
    NNdc_bin = randn(NNdc_outSz,NNdc_hlSz,1);           % input bias              
    NNdc_bout = randn(NNdc_outSz,1,1);                  % output bias
    NNdc_Wout = -XavierUniR2+2*XavierUniR2*rand(NNdc_outSz,1,NNdc_hlSz);          % output weights
    
    %% Total Network
    netDim = [C1_inSz NNdc_hlSz C1_outSz D2_inSz NNdc_hlSz D2_outSz NNdc_inSz NNdc_hlSz NNdc_outSz];
    learnables = [reshape(C1_a,1,[]) reshape(D2_c,1,[]) ];  %% for transformation
    % for NN3 write Win, bin, Wout, bout PER OUTPUT i.e., Win(1), bin(1), Wout(1), bout(1), Win(2), bin(2), ...
    for i = 1:NNdc_outSz
        learnables = [learnables reshape(squeeze(NNdc_Win(i,:,:)),1,[]) reshape(squeeze(NNdc_bin(i,:,:)),1,[]) ...
            reshape(squeeze(NNdc_Wout(i,:,:)),1,[]) reshape(squeeze(NNdc_bout(i,:,:)),1,[])]; 
    end
end

%% function to unravel learnable parameters per NN requested
% C and D are linear transformations and NN is a network
function [Win, bin, Wout, bout] = unravelLearn(learnables,netDim, fNN)
    % netDim carries the dimension of the NN in TRIPLES (INSZ, HLSZ, OUTSZ)
    if fNN==1
        C1_inSz = netDim(1);
        C1_outSz = netDim(3);
        %% unravel learnables of NN3
        % the first MxN correspond to the 1st transformation 
        dummy = learnables(1,1:C1_outSz*C1_inSz);
        C1_a = reshape(dummy,[C1_inSz,C1_outSz])';
        % pass it as Win
        Win = C1_a;
        dummy(1:C1_outSz*C1_inSz) = [];
        if ~isempty(dummy); error('Wrong unraveling in unravelLearn function'); end
    elseif fNN==2
        prevLearnSz = netDim(1)*netDim(3);
        D2_inSz = netDim(4);
        D2_outSz = netDim(6);
        %% unravel learnables of NN3
        % the 2nd (M-N)xN correspond to the 2nd transformation 
        dummy = learnables(prevLearnSz+1:prevLearnSz+D2_outSz*D2_inSz);
        D2_c = reshape(dummy,[D2_inSz,D2_outSz])';
        % pass it as Win
        Win = D2_c;
        dummy(1:D2_outSz*D2_inSz) = [];
        if ~isempty(dummy); error('Wrong unraveling in unravelLearn function'); end
    elseif fNN==3
        prevLearnSz = netDim(1)*netDim(3)+netDim(4)*netDim(6);
        NN3_inSz = netDim(7);
        NN3_HLsz = netDim(8);
        NN3_outSz = netDim(9);
        %% unravel learnables of NN3
        % first N^2 are for linear transformation
        dummy = learnables(prevLearnSz+1:end);
        patchSz = numel(dummy)/NN3_outSz;
        Win = zeros(NN3_outSz,NN3_HLsz,NN3_inSz);
        bin = zeros(NN3_outSz,NN3_HLsz,1);
        Wout = zeros(NN3_outSz,1,NN3_HLsz);
        bout = zeros(NN3_outSz,1,1);
        for i = 1:NN3_outSz         %% number of patches: M
            k = patchSz*(i-1);
            Win(i,:,:) = reshape(dummy(1:NN3_inSz*NN3_HLsz),[NN3_HLsz, NN3_inSz]);
            dummy(1:NN3_inSz*NN3_HLsz) = [];
            bin(i,:,1) = reshape(dummy(1:NN3_HLsz),[NN3_HLsz, 1]);
            dummy(1:NN3_HLsz) = [];
            Wout(i,1,:) = reshape(dummy(1:NN3_HLsz),[1, NN3_HLsz]);
            dummy(1:NN3_HLsz) = [];
            bout(i,1,1) = reshape(dummy(1:1),[1, 1]);
            dummy(1) = [];
        end
        if ~isempty(dummy); error('Wrong unraveling in unravelLearn function'); end
    end

end

%% activation function
%
% seperate so you can change it wherever
function [s, dsdx, ddsdx] = activationFun(x)
    s = logsig(x);
    if nargout>1
        dsdx = s.*(1-s);
        if nargout>2
            ddsdx = s.*(1-s).*(1-2*s);
        end
    end
end