%% Script to generate the synthetic data of the fCSI mechanism, considered using the state variables
clear
clc

rng default; % for reproducibility

%% set dimensions and flags
nSpecs = 4;   
nSamplesPT = 500;
nSamplesTrain = 700;                    % 178 learnables here. (similar to TMDD and MM data/learnables ratio)
nSamplesTest = 5^nSpecs*100;            % number of points of test sets
fTrainData = true;                      % if true prepares the Training/Validation data
fTestData = true;                       % if true prepares the Testing data

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

%% set domain to approximate the SIM in
DomC = [1e-5 80; 1e-5 30; 1e-3 80; 1e-3 30];

%%%%%%%% TRAIN DATA PREPARATION %%%%%%%%%%
if fTrainData
    %% multiple ICs: 5 per Spec
    nICs = 5^nSpecs;
    y_init = rand(nSpecs,nICs);
    y_init(1,:) = 50 + 101*y_init(1,:);   % s10Bar = [50, 151]
    y_init(3,:) = 50 + 101*y_init(3,:);   % s20Bar = [50, 151]   
    % c10, c20 = [0, 1]
    allData = zeros(nSpecs+1,nICs*nSamplesPT);
    for i = 1:nICs
        parVec = [k1f k1b k2 k3f k3b k4 e0 y_init(1,i) y_init(2,i) y_init(3,i) y_init(4,i)];
        yAll = getINHsolGRID(parVec,nSpecs,nSamplesPT,1e-5);
        allData(:,(i-1)*nSamplesPT+1:i*nSamplesPT) = yAll; 
    end
  
    %% Cut data in desired domain Omega
    for i=1:nSpecs
        inDomC = allData(i+1,:)>=DomC(i,1) & allData(i+1,:)<=DomC(i,2);
        allData = allData(:,inDomC);
    end

    save CompInh_TrainData dataTrain;
    save CompInh_allData allData;

else
    load CompInh_TrainData dataTrain;
    load CompInh_allData allData;
end

%%%%%%%% TEST DATA PREPARATION %%%%%%%%%%
if fTestData
    %% multiple ICs: 5 per Spec
    nICs = 5^nSpecs;
    y_init = rand(nSpecs,nICs);
    y_init(1,:) = 50 + 101*y_init(1,:);   % s10Bar = [50, 151]
    y_init(3,:) = 50 + 101*y_init(3,:);   % s20Bar = [50, 151]   
    % c10, c20 = [0, 1]
    allData = zeros(nSpecs+1,nICs*nSamplesPT);
    for i = 1:nICs
        parVec = [k1f k1b k2 k3f k3b k4 e0 y_init(1,i) y_init(2,i) y_init(3,i) y_init(4,i)];
        yAll = getINHsolGRID(parVec,nSpecs,nSamplesPT,1e-5);
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

    save CompInh_TestData allData;
else
    load CompInh_TestData allData;
end

return