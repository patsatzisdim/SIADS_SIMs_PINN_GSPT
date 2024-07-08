%% script to learn PINN for SIM approximation and compare with GSPT-derived ones
clear
clc

%rng default; % for reproducibility

%% set dimensions before calling the generating function
inSz = 3;   % number of input vars z = (u,v) 
outSz = 3;  % number of output vars (x, y) 
Mfast = 1;  % number of assumed fast vars: x
noHL = 1;   % number of hidden layers
hlSz = 20;   % and neurons on them
firstTrain = true;
RPnoRuns = 1; % number of random runs

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DATA PREPARATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% set the parameters for full model
kon = 0.091;
koff = 0.001;
kel = 0.0015;
ksyn = 0.11;
kdeg = 0.0089;
kint = 0.003;

load TMDDP4_TrainData dataTrain;
load TMDDP4_TestData allData;

pars = [kon koff kel ksyn kdeg kint];
z2sol = 1; % corresponding to L
fImpNR = true;  
[QSSAL_SIM, QSSAR_SIM,  QSSARL_SIM, PEA1_SIM, CSPL11_SIM, CSPL21_SIM, ...
        CSPR11_SIM, CSPR21_SIM, CSPRL11_SIM, CSPRL21_SIM] = TMDD_knownSIMs(allData(2:4,:),pars,z2sol,fImpNR);

%% many runs to get averages
CPUrecs = zeros(RPnoRuns,1);
trainMSE = zeros(RPnoRuns,1);
CV_MSE = zeros(RPnoRuns,1);
trainLinf = zeros(RPnoRuns,1);
CV_Linf = zeros(RPnoRuns,1);
trainL2 = zeros(RPnoRuns,1);
CV_L2 = zeros(RPnoRuns,1);
learnedAll = zeros(RPnoRuns,inSz^2+Mfast*(hlSz*(inSz-Mfast+2)+1));
for i = 1:RPnoRuns
    %% Form Data Sets
    XData = dataTrain(end-outSz+1:end,:);              % solution of ODE system
    % random selection of test set
    idx = randperm(size(XData,2),floor(size(XData,2)*0.2));
    Xtrain_CV = XData(:,idx);
    Xtrain = XData;
    Xtrain(:,idx) = [];

    %% precompute RHS to save time from TMDDode F(z)=dzdt
    Fz = TMDDode(0., Xtrain, kon,koff,kel,ksyn,kdeg,kint); % autonomous system

    tic
    %% set architecture and initialize NNs
    [learnables, netDim] = prepareNetwork(inSz,noHL,hlSz,outSz,Mfast,Xtrain);

    %% Train PIML
    if firstTrain
        %% Initial train to configure Transformation
        options=optimoptions(@lsqnonlin,'Display','none','MaxIter',300,'TolFun',1e-6,...
            'Algorithm','levenberg-marquardt','MaxFunEvals',1000000,'TolX',1e-4,...
            'UseParallel',false,'SpecifyObjectiveGradient',true);%,'FinDiffRelStep',1e-6,'CheckGradients',true,'FiniteDifferenceType','central');
        [learned1,resnorm1,RES1,fExit1,Out1] = lsqnonlin(@(curLearn) funPIloss(curLearn,netDim,Xtrain,Fz,2),learnables,[],[],options);
        %% If not determined with accuracy, use weight
        if abs(sum(learned1(1,1:inSz^2))-inSz)>1e-5
            options.MaxIterations = 50;
            [learned2,resnorm2,RES2,fExit2,Out2] = lsqnonlin(@(curLearn) funPIloss(curLearn,netDim,Xtrain,Fz,1),learned1,[],[],options);
        else
            learned2 = learned1;
        end
        %% Round the Transformation and optimize the NN
        TransLearned1 = learned2(1,1:inSz^2);      % reshape is per columns
        TransLearned = round(TransLearned1,2);  % round it to 1e-3;
        if (sum(TransLearned) >= inSz-1e-5) || (sum(TransLearned) <= inSz+1e-5)
            learnables = learned2(1,inSz^2+1:end);
            options.MaxIterations = 500;
            options.TolFun = 1e-10;
            options.TolX = 1e-8;
            [learned,resnorm,RES,fExit,Out] = lsqnonlin(@(curLearn) funPIloss(curLearn,netDim,Xtrain,Fz,3,TransLearned),...
                learnables,[],[],options);
        else
            learned = learned2(1,inSz^2+1:end);
            TransLearned;
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
    trainMSE(i,1) = mse(trainSIM,XzTr);             %% X(z)-h(Y(z))
    trainLinf(i,1) = norm(trainSIM-XzTr,Inf);
    trainL2(i,1) = norm(trainSIM-XzTr,2);
    XzCV = Xtrans(Xtrain_CV,learned,netDim);
    YzCV = Ytrans(Xtrain_CV,learned,netDim);
    CV_SIM = forwardNN3(YzCV,learned,netDim);
    CV_MSE(i,1) = mse(CV_SIM,XzCV);   
    CV_Linf(i,1) = norm(CV_SIM-XzCV,Inf);
    CV_L2(i,1) = norm(CV_SIM-XzCV,2);
end

%% mean, std and Confidence intervals of training assuming no normal distrinution
muCPU = mean(CPUrecs); stdCPU = std(CPUrecs);
muT2 = mean(trainL2); stdT2 = std(trainL2);
muTinf = mean(trainLinf); stdTinf = std(trainLinf);
muTmse = mean(trainMSE); stdTmse = std(trainMSE);
muV2 = mean(CV_L2); stdV2 = std(CV_L2);
muVinf = mean(CV_Linf); stdVinf = std(CV_Linf);
muVmse = mean(CV_MSE); stdVmse = std(CV_MSE);
runs4CIs = [floor(0.02*RPnoRuns) floor(0.05*RPnoRuns) floor(0.1*RPnoRuns) floor(0.9*RPnoRuns) floor(0.95*RPnoRuns) floor(0.98*RPnoRuns)];
CPUrecsS = sort(CPUrecs,'ascend');
trainL2s = sort(trainL2,'ascend');
trainLinfs = sort(trainLinf,'ascend');
trainMSEs = sort(trainMSE,'ascend');
CV_L2s = sort(CV_L2,'ascend');
CV_Linfs = sort(CV_Linf,'ascend');
CV_MSEs = sort(CV_MSE,'ascend');
ciCPU = CPUrecsS(runs4CIs);
ciT2 = trainL2s(runs4CIs); ciTinf = trainLinfs(runs4CIs); ciTmse = trainMSEs(runs4CIs);
ciV2 = CV_L2s(runs4CIs); ciVinf = CV_Linfs(runs4CIs); ciVmse = CV_MSEs(runs4CIs);

%% Metrics on training
fprintf('-------Training metrics--------\n')
fprintf('CPU times:           mean             std             5-95 CI   \n');
fprintf('               %e      %e      %e   %e \n',muCPU,stdCPU,ciCPU(2),ciCPU(5));
fprintf('Errors (L2):         mean             std             5-95 CI   \n');
fprintf('Train Set:     %e      %e      %e   %e \n',muT2,stdT2,ciT2(2),ciT2(5));
fprintf('CV Set:        %e      %e      %e   %e \n',muV2,stdV2,ciV2(2),ciV2(5));
fprintf('Errors (Linf):       mean             std             5-95 CI   \n');
fprintf('Train Set:     %e      %e      %e   %e \n',muTinf,stdTinf,ciTinf(2),ciTinf(5));
fprintf('CV Set:        %e      %e      %e   %e \n',muVinf,stdVinf,ciVinf(2),ciVinf(5));
fprintf('Errors (MSE):        mean             std             5-95 CI   \n');
fprintf('Train Set:     %e      %e      %e   %e \n',muTmse,stdTmse,ciTmse(2),ciTmse(5));
fprintf('CV Set:        %e      %e      %e   %e \n',muVmse,stdVmse,ciVmse(2),ciVmse(5));

Tmetrics = array2table([muCPU stdCPU ciCPU'; muT2 stdT2 ciT2'; muTinf stdTinf ciTinf';...
    muTmse stdTmse ciTmse'; muV2 stdV2 ciV2'; muVinf stdVinf ciVinf'; muVmse stdVmse ciVmse'],...
    "VariableNames",{'mean','std','CI 2','CI 5','CI 10','CI 90','CI 95','CI 98'},...
    "RowNames",{'CPU','TrainL2','TrainLinf','TrainLmse','ValidL2','ValidLinf','ValidLmse'});
save TMDDP4_trainMetric Tmetrics;

%% Test Set errors 
%% PIML
[~,idx] = min(trainMSE);
bestLearned = learnedAll(idx,:);
Xz = Xtrans(allData(2:4,:),bestLearned,netDim);
Yz = Ytrans(allData(2:4,:),bestLearned,netDim);
PIML_SIM = forwardNN3(Yz,bestLearned,netDim);
PIML_MSE = mse(PIML_SIM,Xz);             %% X(z)-h(Y(z))
PIML_Linf = norm(PIML_SIM-Xz,Inf);
PIML_L2 = norm(PIML_SIM-Xz,2);
%% GSPT expressions, HERE all expressions are solved for z1 (explicit or implicit)
z2All = allData(1 + z2sol,:);
QSSAL_MSE = mse(QSSAL_SIM,z2All);                             %% z2_QSSAL - z2
QSSAR_MSE = mse(QSSAR_SIM,z2All);                             %% z2_QSSAR - z2
QSSARL_MSE = mse(QSSARL_SIM,z2All);                           %% z2_QSSARL - z2
PEA_MSE = mse(PEA1_SIM,z2All);                                %% z2_PEA - z2
CSPL11_MSE = mse(CSPL11_SIM,z2All);                           %% z2_CPSL11 - z2
CSPR11_MSE = mse(CSPR11_SIM,z2All);                           %% z2_CPSR11 - z2
CSPRL11_MSE = mse(CSPRL11_SIM,z2All);                         %% z2_CPSRL11 - z2
if ~fImpNR           %% implicit errors 
    CSPL21_MSE = mse(CSPL21_SIM,zeros(size(CSPL21_SIM)));     %% h_CSPL21(z1,z2,z3)-0
    CSPR21_MSE = mse(CSPR21_SIM,zeros(size(CSPR21_SIM)));     %% h_CSPR21(z1,z2,z3)-0
    CSPRL21_MSE = mse(CSPRL21_SIM,zeros(size(CSPRL21_SIM)));  %% h_CSPRL21(z1,z2,z3)-0
else                 %% explicit with NR
    CSPL21_MSE = mse(CSPL21_SIM,z2All);
    CSPR21_MSE = mse(CSPR21_SIM,z2All);
    CSPRL21_MSE = mse(CSPRL21_SIM,z2All);
end
QSSAL_Linf = norm(QSSAL_SIM-z2All,Inf);
QSSAR_Linf = norm(QSSAR_SIM-z2All,Inf);
QSSARL_Linf = norm(QSSARL_SIM-z2All,Inf);
PEA_Linf = norm(PEA1_SIM-z2All,Inf);
CSPL11_Linf = norm(CSPL11_SIM-z2All,Inf);
CSPR11_Linf = norm(CSPR11_SIM-z2All,Inf);
CSPRL11_Linf = norm(CSPRL11_SIM-z2All,Inf);
if ~fImpNR
    CSPL21_Linf = norm(CSPL21_SIM,Inf);
    CSPR21_Linf = norm(CSPR21_SIM,Inf);
    CSPRL21_Linf = norm(CSPRL21_SIM,Inf);
else
    CSPL21_Linf = norm(CSPL21_SIM-z2All,Inf);
    CSPR21_Linf = norm(CSPR21_SIM-z2All,Inf);
    CSPRL21_Linf = norm(CSPRL21_SIM-z2All,Inf);
end
QSSAL_L2 = norm(QSSAL_SIM-z2All,2);
QSSAR_L2 = norm(QSSAR_SIM-z2All,2);
QSSARL_L2 = norm(QSSARL_SIM-z2All,2);
PEA_L2 = norm(PEA1_SIM-z2All,2);
CSPL11_L2 = norm(CSPL11_SIM-z2All,2);
CSPR11_L2 = norm(CSPR11_SIM-z2All,2);
CSPRL11_L2 = norm(CSPRL11_SIM-z2All,2);
if ~fImpNR
    CSPL21_L2 = norm(CSPL21_SIM,2);
    CSPR21_L2 = norm(CSPR21_SIM,2);
    CSPRL21_L2 = norm(CSPRL21_SIM,2);
else
    CSPL21_L2 = norm(CSPL21_SIM-z2All,2);
    CSPR21_L2 = norm(CSPR21_SIM-z2All,2);
    CSPRL21_L2 = norm(CSPRL21_SIM-z2All,2);
end
fprintf('Test Set errors: on data of SIM \n')
fprintf('L2  :   PIML(e)     QSSAL(e)     QSSAR(e)     QSSARL(e)     PEAc(e)     CSPL11(e)    CSPL21(i)    CSPR11(e)    CSPR21(i)    CSPRL11(e)    CSPRL21(i)\n');
fprintf('     %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   \n',PIML_L2,QSSAL_L2,QSSAR_L2,QSSARL_L2,PEA_L2,CSPL11_L2,CSPL21_L2,CSPR11_L2,CSPR21_L2,CSPRL11_L2,CSPRL21_L2);
fprintf('Linf:    \n');
fprintf('     %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   \n',PIML_Linf,QSSAL_Linf,QSSAR_Linf,QSSARL_Linf,PEA_Linf,CSPL11_Linf,CSPL21_Linf,CSPR11_Linf,CSPR21_Linf,CSPRL11_Linf,CSPRL21_Linf);
fprintf('MSE :    \n');
fprintf('     %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   \n',PIML_MSE,QSSAL_MSE,QSSAR_MSE,QSSARL_MSE,PEA_MSE,CSPL11_MSE,CSPL21_MSE,CSPR11_MSE,CSPR21_MSE,CSPRL11_MSE,CSPRL21_MSE);
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
    TransSumR = sum(AugTr,2)-1;    % a+c=1, b+d=1
    TransSumCF = sum((AugTr(1:netDim(3),:)),1);    % sum of Columns for fast vars
    TransSumCS = sum((AugTr(netDim(3)+1:end,:)),1);    % sum of Columns for slow vars
    TransSumC = TransSumCF+TransSumCS-1; % sum of columns is 1
    TransMultCS1 = TransSumCF.*AugTr(netDim(3)+1:end,:);  % totally separated fast and slow
    TransMultCS2 = TransSumCS.*AugTr(1:netDim(3),:);
    %% collection of optimization function with pinning conditions
    if fDim==1
        weight = 100;
        TotMin = [reshape(FminSIM,[],1) ; reshape(FminIE,[],1); ...      %% column per m, WATCH OUT
            reshape(weight*TransSumC,[],1) ; reshape(weight*TransSumR,[],1) ; ...
            reshape(weight*TransMultCS1',[],1); reshape(weight*TransMultCS2',[],1)];
    elseif fDim==2
        weight = 1;
        TotMin = [reshape(FminSIM',[],1) ; reshape(FminIE',[],1) ; ...    %% column per m, WATCH OUT
            reshape(weight*TransSumC,[],1) ; reshape(weight*TransSumR,[],1) ; ...
            reshape(weight*TransMultCS1',[],1); reshape(weight*TransMultCS2',[],1)];
    elseif fDim==3
        TotMin = [reshape(FminSIM,[],1) ; reshape(FminIE,[],1)];
        if (sum(abs(TransSumC))~=0) || (sum(abs(TransSumR))~=0) || (sum(abs(TransMultCS1),'all')~=0) || (sum(abs(TransMultCS2),'all')~=0)
            error('Smth went wrong with the roundoff of the Transformation');
        end
    end

    if nargout > 1
        sz1 = size(FminSIM,2);
        Mf = netDim(3);
        Ntot = netDim(1);
        if fDim == 2
            Jac_shift = 0;
        elseif fDim == 3
            Jac_shift = -Ntot^2;
        end
        Jac = zeros(size(TotMin,1),size(curLearn,2)+Jac_shift);

        %% FminSIM ders
        if fDim == 2
            % Xz trans ders
            for m = 1:Mf
                Jac(1:sz1,(m-1)*Ntot+1:m*Ntot) = UVin';
            end
            % Yz trans ders
            for nmm = 1:Ntot-Mf
                Jac(1:sz1,Mf*Ntot +(nmm-1)*Ntot+1:Mf*Ntot +nmm*Ntot) = -(squeeze(dNdy(1:Mf,nmm,:))'.*UVin)'; % MF is 1 here so different syntax
            end
        end
        %% Get all NN derivatives (used also next)
        % NN pars ders, for Mf = 1
        [dNNdWo, ddNNdWody, dNNdBo, ddNNdBody, dNNdW, ddNNdWdy, dNNdB, ddNNdBdy] = gradsNN(Yz, curLearn, netDim);
        % Win, Bin, Wout, Bout ders
        temp1 = [dNNdW' dNNdB' dNNdWo' dNNdBo'];
        Jac(1:sz1,Jac_shift+Ntot^2+1:end) = -temp1;

        %% FminIE ders
        sz2 = size(FminIE,2);
        if fDim == 2
            % Xz trans ders
            for m = 1:Mf
                Jac(sz1+1:sz1+sz2,(m-1)*Ntot+1:m*Ntot) = Fz';
            end
            % Yz trans ders
            % here M = 1 so no need for loop on m
            for nmm = 1:Ntot-Mf  % fix d
                dNdyd = squeeze(dNdy(1,nmm,:))';
                for j = 1:Ntot   % fix j
                    ddNm_dyh_dDdj = squeeze(ddNNdy(1,:,nmm,:)).*UVin(j,:);
                    Jac(sz1+1:sz1+sz2,Mf*Ntot + (nmm-1)*Ntot+j) = - (dNdyd.*Fz(j,:))' - (sum((AugTr(Ntot-Mf:Ntot,:)'*ddNm_dyh_dDdj).*Fz,1))';
                end
            end
        end
        % NN pars ders  
        temp2 = [ddNNdWdy; ddNNdBdy; ddNNdWody; ddNNdBody];
        for j = 1:Ntot
            sum1 = zeros(sz2,size(Jac,2)-Ntot^2-Jac_shift);
            for nmm = 1:Ntot-Mf
                sum1 = sum1 + (squeeze(temp2(:,nmm,:))*AugTr(Mf+nmm,j))';
            end
            Jac(sz1+1:sz1+sz2,Jac_shift+Ntot^2+1:end) = Jac(sz1+1:sz1+sz2,Jac_shift+Ntot^2+1:end) - sum1.*Fz(j,:)';
        end

        %% Transformation ders
        if fDim == 2
            % sum of cols - 1
            for m = 1:Mf
                Jac(sz1+sz2+1:sz1+sz2+Ntot,(m-1)*Ntot+1:m*Ntot) = eye(Ntot);
            end
            for nmm = 1:Ntot-Mf
                Jac(sz1+sz2+1:sz1+sz2+Ntot,Mf*Ntot +(nmm-1)*Ntot+1:Mf*Ntot +nmm*Ntot) = eye(Ntot);
            end
            % sum of rows -1 
            for m = 1:Mf
                Jac(sz1+sz2+Ntot+1:sz1+sz2+Ntot+m,(m-1)*Ntot+1:m*Ntot) = ones(1,Ntot); 
            end
            for nmm = 1:Ntot-Mf
                Jac(sz1+sz2+Ntot+Mf+nmm,Mf*Ntot +(nmm-1)*Ntot+1:Mf*Ntot +nmm*Ntot) = ones(1,Ntot);
            end
            % sum of fast * every slow, per j = 1:Ntot i.e. per blocks of Ntot (verticaly and horizontaly)
            for nmm = 1:Ntot-Mf
                Jac(sz1+sz2+2*Ntot+(nmm-1)*Ntot+1:sz1+sz2+2*Ntot+nmm*Ntot,1:Ntot*Mf) = repmat(diag(AugTr(Mf+nmm,:)),1,Mf);
                Jac(sz1+sz2+2*Ntot+(nmm-1)*Ntot+1:sz1+sz2+2*Ntot+nmm*Ntot,Ntot*Mf+(nmm-1)*Ntot+1:Ntot*Mf+nmm*Ntot) = ...
                    diag(TransSumCF);
            end
            % sum of slow * every fast, per j=1:nTot, i.e., per blocs of Ntot (ver and hor)
            for m = 1:Mf
                Jac(sz1+sz2+2*Ntot+(Ntot-Mf)*Ntot+(m-1)*Ntot+1:sz1+sz2+2*Ntot+(Ntot-Mf)*Ntot+m*Ntot,(m-1)*Ntot+...
                    1:m*Ntot) = diag(TransSumCS); 
                Jac(sz1+sz2+2*Ntot+(Ntot-Mf)*Ntot+(m-1)*Ntot+1:sz1+sz2+2*Ntot+(Ntot-Mf)*Ntot+m*Ntot,Ntot*Mf+...
                    1:Ntot^2) = repmat(diag(AugTr(m,:)),1,Ntot-Mf);
            end 
            Jac(Mf*(sz1+sz2)+1:Mf*(sz1+sz2)+2*Ntot+Ntot^2,:) = weight*Jac(Mf*(sz1+sz2)+1:Mf*(sz1+sz2)+2*Ntot+Ntot^2,:);
        end
    end

end

% function to calculate the gradients of the NN w.r.t. parameters
function [dNNdWo, ddNNdWody, dNNdBo, ddNNdBody, dNNdW, ddNNdWdy, dNNdB, ddNNdBdy] = gradsNN(Yin, learnables, netDim)
    [Win, bin, Wout, bout] = unravelLearn(learnables,netDim,3);
    hlSz = size(Win,1);
    inSz = size(Win,2);
    Ns = size(Yin,2);
    %
    [phiL, dphiL, ~] = activationFun(Win*Yin+bin);
    dNNdWo = phiL;                       % per i,j dim: M x ML
    ddNNdWody = zeros(hlSz,inSz,Ns);            % L x M x (N-M) x Ns (M=1 omitted)
    for i=1:inSz
        ddNNdWody(:,i,:) = Win(:,i).*dphiL;    
    end
    %
    dNNdB = Wout'.*dphiL;                                       
    ddNNdBdy = zeros(hlSz,inSz,Ns);                             
    for i=1:inSz
        ddNNdBdy(:,i,:) = dNNdB.*Win(:,i).*(1-2*phiL);   
    end
    %ddNNdBdy = Wout'.*Win(:,1).*phiL.*(1-phiL).*(1-2*phiL);
    %
    dNNdBo = ones(1,size(Yin,2));
    ddNNdBody = zeros(1,inSz,size(Yin,2));
    %
    dNNdW = zeros(inSz*hlSz,Ns);                                     % per i,j dim: M x MDL, that is both N-M and epsilon
    ddNNdWdy = zeros(inSz*hlSz,inSz,Ns);                             % per i,j dim: M x MDL x d for d=1...N-M
    for j=1:inSz       
        %dNNdW1 = Wout'.*dphiL.*Yin(i,:);                            
        dNNdW(hlSz*(j-1)+1:hlSz*j,:) = dNNdB.*Yin(j,:); 
        for i=1:inSz   % i samples d=1,N-M
            if i==j  % when d=h
                %ddNNdWdy1 = Wout'.*dphiL.*(Win(:,i).*(1-2*phiL).*Yin(j,:)+1);   % per i,j dim: M x M D L x d
                ddNNdWdy(hlSz*(j-1)+1:hlSz*j,i,:) = dNNdB.*(Win(:,i).*(1-2*phiL).*Yin(j,:)+1);
            else    % when d~=h 
                ddNNdWdy(hlSz*(j-1)+1:hlSz*j,i,:) = dNNdB.*(Win(:,i).*(1-2*phiL).*Yin(j,:));
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
    [phi, dphi, ddphi] = activationFun(Win*Yin+bin);
    NN3_out = Wout*phi+bout;
    if nargout>1
        dNN3_dy = zeros(netDim(9),netDim(7),size(Yin,2)); %% dhdy is Mx(N-M) for each input point
        for i = 1:size(Yin,2)
            dNN3_dy(:,:,i) = Wout*(Win.*dphi(:,i));
        end
        if nargout>2
            ddNN3_dy = zeros(netDim(9),netDim(7),netDim(7),size(Yin,2));  % d(dhdy)/dy 
            for i = 1:netDim(7)
                for j = 1:netDim(7)
                    ddNN3_dy(1,i,j,:) = Wout*(Win(:,i).*Win(:,j).*ddphi);
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
    NNdc_Win = -XavierUniR1+2*XavierUniR1*rand(NNdc_hlSz,NNdc_inSz);   % input weights
    NNdc_bin = randn(NNdc_hlSz,1);           % input bias             
    NNdc_bout = randn(NNdc_outSz,1);         % output bias
    NNdc_Wout = -XavierUniR2+2*XavierUniR2*rand(NNdc_outSz,NNdc_hlSz); % output weights
    
    %% Total Network
    netDim = [C1_inSz NNdc_hlSz C1_outSz D2_inSz NNdc_hlSz D2_outSz NNdc_inSz NNdc_hlSz NNdc_outSz];
    learnables = [reshape(C1_a,1,[]) reshape(D2_c,1,[])...
        reshape(NNdc_Win,1,[]) reshape(NNdc_bin,1,[]) reshape(NNdc_Wout,1,[]) reshape(NNdc_bout,1,[])];             %% NNdc 
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
        dummy = learnables(1:C1_outSz*C1_inSz);
        C1_a = reshape(dummy,[C1_inSz,C1_outSz]);
        % pass it as Win
        Win = C1_a';
        dummy(1:C1_outSz*C1_inSz) = [];
        if ~isempty(dummy); error('Wrong unraveling in unravelLearn function'); end
    elseif fNN==2
        prevLearnSz = netDim(1)*netDim(3);
        D2_inSz = netDim(4);
        D2_outSz = netDim(6);
        %% unravel learnables of NN3
        % the 2nd (M-N)xN correspond to the 2nd transformation 
        dummy = learnables(prevLearnSz+1:prevLearnSz+D2_outSz*D2_inSz);
        D2_c = reshape(dummy,[D2_inSz,D2_outSz]);
        % pass it as Win
        Win = D2_c';
        dummy(1:D2_outSz*D2_inSz) = [];
        if ~isempty(dummy); error('Wrong unraveling in unravelLearn function'); end
    elseif fNN==3
        prevLearnSz = netDim(1)*netDim(3)+netDim(4)*netDim(6);
        NN3_inSz = netDim(7);
        NN3_HLsz = netDim(8);
        NN3_outSz = netDim(9);
        %% unravel learnables of NN3
        % first 4 are for linear transformation
        dummy = learnables(prevLearnSz+1:end);
        Win = reshape(dummy(1:NN3_inSz*NN3_HLsz),[NN3_HLsz, NN3_inSz]);
        dummy(1:NN3_inSz*NN3_HLsz) = [];
        bin = reshape(dummy(1:NN3_HLsz),[NN3_HLsz, 1]);
        dummy(1:NN3_HLsz) = [];
        Wout = reshape(dummy(1:NN3_HLsz*NN3_outSz),[NN3_outSz, NN3_HLsz]);
        dummy(1:NN3_HLsz*NN3_outSz) = [];
        bout = reshape(dummy(1:NN3_outSz),[NN3_outSz, 1]);
        dummy(1:NN3_outSz) = [];
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