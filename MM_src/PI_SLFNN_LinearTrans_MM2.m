%% script to learn PINN for SIM approximation and compare with GSPT-derived ones
clear
clc

% rng default; % for reproducibility

%% set dimensions before calling the generating function
inSz = 2;   % number of input vars z = (u,v) 
outSz = 2;  % number of output vars (x, y) 
Mfast = 1;  % number of assumed fast vars: x
noHL = 1;   % number of hidden layers
hlSz = 20;   % and neurons on them
firstTrain = true;
RPnoRuns = 100; % number of random runs

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DATA PREPARATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% set the parameters for full model
k1f = 1.;
k1b = 1.;
k2 = 1.e-2;
e0 = 1.e+2;

%% load data
load MM2_TrainData dataTrain;
load MM2_TestData allData;

pars = [k1f k1b k2 e0];
z2sol = 2; % corresponding to s
fImpNR = true;  
[sQSSA_SIM, rQSSA_SIM, PEA_SIM, CSPe_SIM, CSPc11_SIM, CSPc21_SIM, ...
        CSPs11_SIM, CSPs21_SIM] = MM_knownSIMs(allData(2:3,:),pars,z2sol,fImpNR);


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

    %% precompute RHS ders to save time from MMode F(z)=dzdt
    Fz = MMode(0., Xtrain, k1f, k1b, k2, e0); % autonomous system
    %

    tic
    %% set architecture and initialize NNs
    [learnables, netDim] = prepareNetwork(inSz,noHL,hlSz,outSz,Mfast,Xtrain);
    %% Train PIML
    if firstTrain
        %% Initial train to determine the Transformation 
        options=optimoptions(@lsqnonlin,'Display','none','MaxIter',1950,'TolFun',1e-6,...
            'Algorithm','levenberg-marquardt','MaxFunEvals',1000000,'TolX',1e-4,'StepTolerance',1e-8,...
            'UseParallel',false,'SpecifyObjectiveGradient',true);%,'FinDiffRelStep',1e-6,'CheckGradients',true,'FiniteDifferenceType','central');
        [learned1,resnorm1,RES1,fExit1,Out1] = lsqnonlin(@(curLearn) funPIloss(curLearn,netDim,Xtrain,Fz,2),learnables,[],[],options);
        %% If not determined with accuracy, use weight
        if abs(sum(learned1(1,1:4))-inSz)>1e-5
            options.MaxIterations = 50;
            [learned2,resnorm2,RES2,fExit2,Out2] = lsqnonlin(@(curLearn) funPIloss(curLearn,netDim,Xtrain,Fz,1),learned1,[],[],options);
        else
            learned2 = learned1;
        end
        %% Round the Transformation and optimize the NN
        TransLearned1 = learned2(1,1:inSz^2);      % reshape is per columns
        TransLearned = round(TransLearned1,2);  % round it to 1e-3;
        %%
        if (sum(TransLearned) >= inSz-1e-5) && (sum(TransLearned) <= inSz+1e-5)
            learnables = learned2(1,inSz^2+1:end);
            options.MaxIterations = 1000;
            options.TolFun = 1e-9;
            options.TolX = 1e-7;
            [learned,resnorm,RES,fExit,Out] = lsqnonlin(@(curLearn) funPIloss(curLearn,netDim,Xtrain,Fz,3,TransLearned),...
                learnables,[],[],options);
        else
            TransLearned;
        end
        learned = [TransLearned learned];

    end
    CPUend = toc;
    CPUrecs(i,1) = CPUend;

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
ciT2 = muT2 + ts*semT2; ciTinf = muTinf + ts*semTinf; ciTmse = muTmse + ts*semTmse;
ciV2 = muV2 + ts*semV2; ciVinf = muVinf + ts*semVinf; ciVmse = muVmse + ts*semVmse;

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

Tmetrics = array2table([muCPU stdCPU ciCPU; muT2 stdT2 ciT2; muTinf stdTinf ciTinf;...
    muTmse stdTmse ciTmse; muV2 stdV2 ciV2; muVinf stdVinf ciVinf; muVmse stdVmse ciVmse],...
    "VariableNames",{'mean','std','CI 2.5','CI 5.0','CI 10.0','CI 90.0','CI 95.0','CI 97.5'},...
    "RowNames",{'CPU','TrainL2','TrainLinf','TrainLmse','ValidL2','ValidLinf','ValidLmse'});
save MM2_trainMetric Tmetrics;


%% Test Set errors 
%% PIML
[~,idx] = min(CV_MSE);                  % select the one with the most accurate cross-validation error
bestLearned = learnedAll(idx,:);
Xz = Xtrans(allData(2:3,:),bestLearned,netDim);
Yz = Ytrans(allData(2:3,:),bestLearned,netDim);
PIML_SIM = forwardNN3(Yz,bestLearned,netDim);
PIML_MSE = mse(PIML_SIM,Xz);             %% X(z)-h(Y(z))
PIML_Linf = norm(PIML_SIM-Xz,Inf);
PIML_L2 = norm(PIML_SIM-Xz,2);
%% GSPT expressions, HERE all expressions are solved for z2 (explicit or implicit)
z2All = allData(1 + z2sol,:);
sQSSAMSE = mse(sQSSA_SIM,z2All);                             %% z2_sQSSA - z2
rQSSAMSE = mse(rQSSA_SIM,z2All);                             %% z2_rQSSA - z2
PEAcMSE = mse(PEA_SIM,z2All);                                %% z2_PEA - z2
CSPc11MSE = mse(CSPc11_SIM,z2All);                           %% z2_CPS11c - z2
CSPs11MSE = mse(CSPs11_SIM,z2All);                           %% z2_CPS11s - z2
CSPs21MSE = mse(CSPs21_SIM,z2All);                           %% z2_CPS11s - z2
CSPeMSE = mse(CSPe_SIM,z2All);                               %% z2_CPS11s - z2
if ~fImpNR           %% implicit errors 
    CSPc21MSE = mse(CSPc21_SIM,zeros(size(CSPc21_SIM)));     %% h_CSPc21(z1,z2)-0
else                 %% explicit with NR
    CSPc21MSE = mse(CSPc21_SIM,z2All);
end
sQSSALinf = norm(sQSSA_SIM-z2All,Inf);
rQSSALinf = norm(rQSSA_SIM-z2All,Inf); 
PEAcLinf = norm(PEA_SIM-z2All,Inf);
CSPc11Linf = norm(CSPc11_SIM-z2All,Inf);
CSPs11Linf = norm(CSPs11_SIM-z2All,Inf);
CSPs21Linf = norm(CSPs21_SIM-z2All,Inf);
CSPeLinf = norm(CSPe_SIM-z2All,Inf);
if ~fImpNR
    CSPc21Linf = norm(CSPc21_SIM,Inf);
else
    CSPc21Linf = norm(CSPc21_SIM-z2All,Inf);
end
sQSSAL2 = norm(sQSSA_SIM-z2All,2);
rQSSAL2 = norm(rQSSA_SIM-z2All,2);
PEAcL2 = norm(PEA_SIM-z2All,2);
CSPc11L2 = norm(CSPc11_SIM-z2All,2);
CSPs11L2 = norm(CSPs11_SIM-z2All,2);
CSPs21L2 = norm(CSPs21_SIM-z2All,2);
CSPeL2 = norm(CSPe_SIM-z2All,2);
if ~fImpNR
    CSPc21L2 = norm(CSPc21_SIM,2);
else
    CSPc21L2 = norm(CSPc21_SIM-z2All,2);  
end
fprintf('Test Set errors: on data of SIM \n')
fprintf('L2  :   PIML(e)     sQSSA(e)     rQSSA(e)     PEAc(e)      CSPe(i)     CSPc11(e)    CSPc21(i)    CSPs11(e)    CSPs21(i)\n');
fprintf('     %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   \n',PIML_L2,sQSSAL2,rQSSAL2,PEAcL2,CSPeL2,CSPc11L2,CSPc21L2,CSPs11L2,CSPs21L2);
fprintf('Linf:    \n');
fprintf('     %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   \n',PIML_Linf,sQSSALinf,rQSSALinf,PEAcLinf,CSPeLinf,CSPc11Linf,CSPc21Linf,CSPs11Linf,CSPs21Linf);
fprintf('MSE :    \n');
fprintf('     %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e   \n',PIML_MSE,sQSSAMSE,rQSSAMSE,PEAcMSE,CSPeMSE,CSPc11MSE,CSPc21MSE,CSPs11MSE,CSPs21MSE);
if ~fImpNR
    fprintf('Implicit forms are not solved for the fast variable numerically!\n');
else
    fprintf('Implicit forms are solved for the fast variable with Newton numerically!\n');
end


return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Physics-informed loss function 
%    
%  minimization of SIMEq                1) X(z)-H(Y(z)) = X(z) - N(Y(z))
%                  InvEq(grad of SIM)   2) [I, -dHdY]*[dXdz; dYdz]*dzdt = [I -dN_dy]*[dX_dz; dY_dz]*RHS(z) 
%                  Transformation related loss  3) see text
function [TotMin, Jac] = funPIloss(curLearn,netDim,UVin,Fz,fDim,TransLearned)
    if fDim == 3
        if nargin ~= 6; error('Supply the transformation'); end
        curLearn = [TransLearned curLearn];
    end
    %% get Tranformations: z->x and z->y and their derivatives
    [Xz, dXdz] = Xtrans(UVin,curLearn,netDim);
    [Yz, dYdz] = Ytrans(UVin,curLearn,netDim);
    %% get SIMEq: y->x
    [NN3Xy, dNdy, ddNdy] = forwardNN3(Yz,curLearn,netDim);

    %% Form argmins (1): X(z)-H(Y(z))
    FminSIM = Xz-NN3Xy;

    %% Form argmins (2): [I, -dHdY]*[dXdz; dYdz]*F(z)
    FminIE = sum((dXdz-dNdy.*dYdz).*Fz,1);

    %% Transformation conditions: x = X(z) and y=Y(z) are linear combinations 
    AugTr = [dXdz(:,1)'; dYdz(:,1)'];
    TransSumR = sum(AugTr,2)-1;    % a+c=1, b+d=1
    TransSumCF = sum(AugTr(1:netDim(3),:),1);    % sum of Columns for fast vars
    TransSumCS = sum(AugTr(netDim(3)+1:end,:),1);    % sum of Columns for slow vars
    TransSumC = TransSumCF+TransSumCS-1; % sum of columns is 1
    TransMultCS1 = TransSumCF.*AugTr(netDim(3)+1:end,:);  % totally separated fast and slow
    TransMultCS2 = TransSumCS.*AugTr(1:netDim(3),:);
    
    
    %% collection of optimization function with pinning conditions
    if fDim == 1
        weight = 100;
        TotMin = [reshape(FminSIM,[],1) ; reshape(FminIE,[],1); ... 
        reshape(weight*TransSumC,[],1) ; reshape(weight*TransSumR,[],1) ; ...
        reshape(weight*TransMultCS1,[],1); reshape(weight*TransMultCS2,[],1) ];
    elseif fDim == 2
        weight = 1;
        TotMin = [reshape(FminSIM,[],1) ; reshape(FminIE,[],1); ... 
        reshape(weight*TransSumC,[],1) ; reshape(weight*TransSumR,[],1) ; ...
        reshape(weight*TransMultCS1,[],1); reshape(weight*TransMultCS2,[],1) ];
    elseif fDim == 3
        TotMin = [reshape(FminSIM,[],1) ; reshape(FminIE,[],1)];
        if (sum(abs(TransSumC))~=0) || (sum(abs(TransSumR))~=0) || (sum(abs(TransMultCS1))~=0) || (sum(abs(TransMultCS2))~=0)
            error('Smth went wrong with the roundoff of the Transformation');
        end
    end

    if nargout>1
        Mf = netDim(3);
        Ntot = netDim(1);
        % for when Trans is known then do not calculate derivatives!
        if (fDim == 1) || (fDim == 2)
            Jac_shift = 0;
        elseif fDim == 3
            Jac_shift = -Ntot^2;
        end
        Jac = zeros(size(TotMin,1),size(curLearn,2)+Jac_shift);
        %% FminSIM ders
        sz1 = size(FminSIM,2);
        if (fDim == 1) || (fDim == 2)
            % Xz trans ders
            for m = 1:Mf
                Jac(1:sz1,(m-1)*Ntot+1:m*Ntot) = UVin';
            end
            % Yz trans ders
            for nmm = 1:Ntot-Mf
                Jac(1:sz1,Mf*Ntot +(nmm-1)*Ntot+1:Mf*Ntot +nmm*Ntot) = -(dNdy(nmm,:).*UVin)';
            end
        end
        % NN pars ders
        %% get all derivatives
        [dNNdWo, ddNNdWody, dNNdBo, ddNNdBody, dNNdW, ddNNdWdy, dNNdB, ddNNdBdy] = gradsNN(Yz, curLearn, netDim);
        % Win, bin, Wout, bout ders
        temp1 = [dNNdW' dNNdB' dNNdWo' dNNdBo'];
        Jac(1:sz1,Jac_shift+Ntot^2+1:end) = -temp1;

        %% FminIE ders
        sz2 = size(FminIE,2);
        if (fDim == 1) || (fDim == 2)
            % Xz trans ders
            for m = 1:Mf
                Jac(sz1+1:sz1+sz2,(m-1)*Ntot+1:m*Ntot) = Fz';
            end
            % Yz trans ders
            % ddNm_dyd dDhj for m, d and h and for every k
            ddNm_dyd_dDhj = ddNdy.*UVin;        
            for nmm = 1:Ntot-Mf
                Jac(sz1+1:sz1+sz2,Mf*Ntot +(nmm-1)*Ntot+1:Mf*Ntot +nmm*Ntot) = -(dNdy(nmm,:).*Fz)' - ...
                    (ddNm_dyd_dDhj.*sum(dYdz.*Fz,1))'; % should be sum internally but h=d=1
            end
        end
        % NN pars ders
        temp2 = [ddNNdWdy; ddNNdBdy; ddNNdWody; ddNNdBody];
        for i = 1:sz2
            Jac(sz1+i,Jac_shift+Ntot^2+1:end) = -((temp2(:,i)*dYdz(:,i)')*Fz(:,i))';
        end

        %% Transformation ders
        if (fDim == 1) || (fDim == 2)     
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
                Jac(sz1+sz2+Ntot+Mf+1:sz1+sz2+Ntot+Mf+nmm,Mf*Ntot +(nmm-1)*Ntot+1:Mf*Ntot +nmm*Ntot) = ones(1,Ntot);
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
            Jac(sz1+sz2+1:sz1+sz2+2*Ntot+Ntot^2,1:Ntot^2) = Jac(sz1+sz2+1:sz1+sz2+2*Ntot+Ntot^2,1:Ntot^2)*weight;
        end

    end

end

% function to calculate the gradients of the NN w.r.t. parameters
function [dNNdWo, ddNNdWody, dNNdBo, ddNNdBody, dNNdW, ddNNdWdy, dNNdB, ddNNdBdy] = gradsNN(Yin, learnables, netDim)
    [Win, bin, Wout, bout] = unravelLearn(learnables,netDim,3);
    %
    phiL = activationFun(Win*Yin+bin);
    dphiL = phiL.*(1-phiL);
    dNNdWo = phiL;                     % per i,j dim: M x ML
    ddNNdWody = Win(:,1).*dphiL;       % per i,j dim: M x ML x d for d=1...N-M
    %
    dNNdB = Wout'.*dphiL;                                         % per i,j dim: M x ML
    %ddNNdBdy = Wout'.*Win(:,1).*phiL.*(1-phiL).*(1-2*phiL);
    ddNNdBdy = dNNdB.*Win(:,1).*(1-2*phiL);                       % per i,j dim: M x ML x d for d=1...N-M
    %
    dNNdBo = ones(1,size(Yin,2));
    ddNNdBody = zeros(1,size(Yin,2));
    %
    %dNNdW1 = Wout'.*dphiL.*Yin(1,:);                                % per i,j dim: M x M(N-M)L
    dNNdW = dNNdB.*Yin(1,:); 
    %ddNNdWdy1 = Wout'.*dphiL.*(Win(:,1).*(1-2*phiL).*Yin(1,:)+1);   % per i,j dim: M x M(N-M)L x d
    ddNNdWdy = dNNdB.*(Win(:,1).*(1-2*phiL).*Yin(1,:)+1);
      
end

%% function for calculating X(z) output: from z->x=X(z) and derivatives x_z
function [XofZ, dXdz] = Xtrans(UVin,learnables,netDim)
    C1_a = learnables(1,1);
    C1_b = learnables(1,2);
    %% forward the input
    XofZ = C1_a*UVin(1,:)+C1_b*UVin(2,:);
    if nargout>1
        dXdz = zeros(netDim(1),size(UVin,2));
        dXdz(1,:) = C1_a*ones(1,size(UVin,2));
        dXdz(2,:) = C1_b*ones(1,size(UVin,2));
    end
end

%% function for calculating Y(z) output: from z->y=Y(z) and derivatives y_z
function [YofZ, dYdz] = Ytrans(UVin,learnables,netDim)
    D2_c = learnables(1,3);
    D2_d = learnables(1,4);    
    %% forward the input
    YofZ = D2_c*UVin(1,:)+D2_d*UVin(2,:);
    if nargout>1
        dYdz = zeros(netDim(4),size(UVin,2));
        dYdz(1,:) = D2_c*ones(1,size(UVin,2));
        dYdz(2,:) = D2_d*ones(1,size(UVin,2));
    end
end

function [NN3_out, dNN3_dy, ddNN3_dy] = forwardNN3(Yin,learnables,netDim)
    [Win, bin, Wout, bout] = unravelLearn(learnables,netDim,3);
    %% forward the input
    [phi, dphi, ddphi] = activationFun(Win*Yin+bin);
    NN3_out = Wout*phi+bout;
    if nargout>1
        dNN3_dy = zeros(netDim(7),size(Yin,2));
        for i = 1:netDim(7)
            dNN3_dy(i,:) = Wout*(Win(:,i).*dphi);
        end
        if nargout>2
            ddNN3_dy = zeros(netDim(7),size(Yin,2));
            for i = 1:netDim(7)
                ddNN3_dy(i,:) = Wout*(Win(:,i).*Win(:,i).*ddphi);
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

    %% C matrix: input state vector z, output fast variables x
    C1_inSz = inSz;                        % N
    C1_outSz = Mfast;                      % M
    NN1_a = 0.5;
    NN1_b = 0.5;
    
    
    %% D matrix: input state vector z, output slow variables x
    D2_inSz = inSz;                        % N
    D2_outSz = outSz-Mfast;                % N-M
    NN2_c = 0.5;
    NN2_d = 0.5;
    
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
    %
    %% Total Network
    netDim = [C1_inSz NNdc_hlSz C1_outSz D2_inSz NNdc_hlSz D2_outSz NNdc_inSz NNdc_hlSz NNdc_outSz];
    learnables = [NN1_a NN1_b NN2_c NN2_d...
        reshape(NNdc_Win,1,[]) reshape(NNdc_bin,1,[]) reshape(NNdc_Wout,1,[]) reshape(NNdc_bout,1,[])];             %% NNdc 
end


%% function to unravel learnable parameters per NN requested
% C and D are linear transformations and NN is a network
function [Win, bin, Wout, bout] = unravelLearn(learnables,netDim, fNN)
    % netDim carries the dimension of the NN in TRIPLES (INSZ, HLSZ, OUTSZ)
    if fNN==3
        NN3_inSz = netDim(7);
        NN3_HLsz = netDim(8);
        NN3_outSz = netDim(9);
        %% unravel learnables of NN3
        % first 4 are for linear transformation
        dummy = learnables(5:end);
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