%% Script to generate the synthetic data of the TMDD mechanism for the 2 SIMs 
clear
clc

rng default; % for reproducibility

%% set dimensions and flags
nSpecs = 3;   
nSamplesPT = 500;
nSamplesTrain = 700; % 20 learnables here (similar for MM ratio of data/parameters)
nSamplesTest = 5^nSpecs*100;                  % number of points of test sets
fTrainData = true;                            % if true prepares the Training/Validation data
fTestData = true;                             % if true prepares the Testing data

%% set the parameters for full model: MM1 case
fPeriod = 2;  % 2 for P2, 4 for P4
kon = 0.091;
koff = 0.001;
kel = 0.0015;
ksyn = 0.11;
kdeg = 0.0089;
kint = 0.003;
if fPeriod == 2
    DomC = [10 140; 1e-3 0.12; 10 33];
elseif fPeriod == 4
    DomC = [5e-3 5e-2; 6 13; 6 27];
end

%%%%%%%% TRAIN DATA PREPARATION %%%%%%%%%%
if fTrainData
    %% multiple ICs: 5 per Spec
    nICs = 5^nSpecs;
    y_init = rand(nSpecs,nICs);
    y_init(1,:) = 50 + 100*y_init(1,:);   % L0 = [50, 150]
    y_init(2,:) = 10 + 5*y_init(2,:);     % R0 = [10, 15]   given that R* = 12.36 = ksyn/kdeg
    y_init(3,:) = y_init(3,:);            % RL0 = [0, 1]    given that RL*  = 0
    allData = zeros(nSpecs+1,nICs*nSamplesPT);
    for i = 1:nICs
        parVec = [kon koff kel ksyn kdeg kint y_init(1,i) y_init(2,i) y_init(3,i)];
        yAll = getTMDDsolGRID(parVec,nSpecs,nSamplesPT,fPeriod);
        allData(:,(i-1)*nSamplesPT+1:i*nSamplesPT) = yAll; 
    end

    %% Cut data in desired domain Omega
    for i=1:nSpecs
        inDomC = allData(i+1,:)>=DomC(i,1) & allData(i+1,:)<=DomC(i,2);
        allData = allData(:,inDomC);
    end
  
    figure(2);
    if fPeriod == 2
        %scatter3(y_init(1,:),y_init(3,:),y_init(2,:),'ro'); hold on;
        scatter3(allData(2,:),allData(4,:),allData(3,:),'b.'); hold on;
        ind = randperm(size(allData,2),nSamplesTrain);
        dataTrain = allData(:,ind);
        scatter3(dataTrain(2,:),dataTrain(4,:),dataTrain(3,:),'ro'); hold off;
    elseif fPeriod == 4
        scatter3(allData(3,:),allData(4,:),allData(2,:),'b.'); hold on;
        ind = randperm(size(allData,2),nSamplesTrain);
        dataTrain = allData(:,ind);
        scatter3(dataTrain(3,:),dataTrain(4,:),dataTrain(2,:),'ro'); hold off;
    end
    
    if fPeriod == 2
        save TMDDP2_TrainData dataTrain;
        save TMDDP2_allData allData;
    elseif fPeriod == 4
        save TMDDP4_TrainData dataTrain;
        save TMDDP4_allData allData;
    end
else
    if fPeriod == 2
        load TMDDP2_TrainData dataTrain;
        load TMDDP2_allData allData;
    elseif fPeriod == 4
        load TMDDP4_TrainData dataTrain;
        load TMDDP4_allData allData;
    end
end

%%%%%%%% TEST DATA PREPARATION %%%%%%%%%%
if fTestData
    %% multiple ICs: 5 per Spec
    nICs = 5^nSpecs;
    nSamplesPT = 100;                     % keep lesser points for enabling visualization
    y_init = rand(nSpecs,nICs);
    y_init(1,:) = 50 + 100*y_init(1,:);   % L0 = [50, 150]
    y_init(2,:) = 10 + 5*y_init(2,:);     % R0 = [10, 15]   given that R* = 12.36 = ksyn/kdeg
    y_init(3,:) = y_init(3,:);            % RL0 = [0, 1]    given that RL*  = 0
    allData = zeros(nSpecs+1,nICs*nSamplesPT);
    for i = 1:nICs
        parVec = [kon koff kel ksyn kdeg kint y_init(1,i) y_init(2,i) y_init(3,i)];
        yAll = getTMDDsolGRID(parVec,nSpecs,nSamplesPT,fPeriod);
        allData(:,(i-1)*nSamplesPT+1:i*nSamplesPT) = yAll; 
    end

    %% Cut data in desired domain Omega
    for i=1:nSpecs
        inDomC = allData(i+1,:)>=DomC(i,1) & allData(i+1,:)<=DomC(i,2);
        allData = allData(:,inDomC);
    end
    
    %% randomly select the test set
    ind = randperm(size(allData,2),nSamplesTest);
    allData = allData(:,ind);

    if fPeriod == 2
        save TMDDP2_TestData allData;
    elseif fPeriod == 4
        save TMDDP4_TestData allData;
    end
else
    if fPeriod == 2
        load TMDDP2_TestData allData;
    elseif fPeriod == 4
        load TMDDP4_TestData allData;
    end
end

return