%% Script to generate the synthetic data of the MM mechanism for the 3 parameter sets considered
clear
clc

rng default; % for reproducibility

%% set dimensions and flags
fMMcase = 3;
nSpecs = 2;   
nSamplesPT = 500;
nSamplesTrain = 500; 
nSamplesTest = 5^nSpecs*100;      % number of points of test sets
fTrainData = true;                % if true prepares the Training/Validation data
fTestData = true;                 % if true prepares the Testing data

%% set the parameters for full model and the domain to collect collocation points!
if fMMcase == 1
    k1f = 1.;
    k1b = 100.;
    k2 = 1.;
    e0 = 1.;
    DomC = [1e-5 1; 1e-3 1e+3];              % domain Omega to collect collocation points
elseif fMMcase == 2
    k1f = 1.;
    k1b = 1.;
    k2 = 1.e-2;
    e0 = 1.e+2;
    DomC = [1e-3 50; 1e-5 1];                % domain Omega to collect collocation points
elseif fMMcase == 3
    k1f = 1.;
    k1b = 1.;
    k2 = 1e+3;
    e0 = 1e+1;
    DomC = [1e-5 1; 1e-3 1e+2];              % domain Omega to collect collocation points
end

%%%%%%%% TRAIN DATA PREPARATION %%%%%%%%%%
if fTrainData
    %% multiple ICs: 5 per Spec
    nICs = 5^nSpecs;
    y_init = rand(2,nICs);
    if fMMcase == 1
        y_init(1,:) = 1e+3 + 1e+2*y_init(1,:);   %% MM1 case
    elseif fMMcase == 2
        y_init(2,:) = 0.5*e0+y_init(2,:);        %% MM2 case
    elseif fMMcase == 3
        y_init(1,:) = 1e+2 + 1e+1*y_init(1,:);   %% MM3 case: k2>k1b+k1f*s (so that 2nd is fast)
        y_init(2,:) = e0 - 0.1*e0*y_init(2,:);   %% MM3 case
    end
    allData = zeros(nSpecs+1,nICs*nSamplesPT);
    for i = 1:nICs
        parVec = [k1f k1b k2 e0 y_init(1,i) y_init(2,i)];
        yAll = getMMsolGRID(parVec,nSpecs,nSamplesPT,min(DomC,[],'all'));
        allData(:,(i-1)*nSamplesPT+1:i*nSamplesPT) = yAll;   
    end

    %% Cut data in desired domain Omega
    for i=1:nSpecs
        inDomC = allData(i+1,:)>=DomC(i,1) & allData(i+1,:)<=DomC(i,2);
        allData = allData(:,inDomC);
    end
   
    %% visualize and collect collocation points
    figure(2);
    scatter(y_init(2,:),y_init(1,:),'ro'); hold on;
    scatter(allData(2,:),allData(3,:),'b.'); hold off;
    set(gca,'YScale','log')
    set(gca,'XScale','log')
    ind = randperm(size(allData,2),nSamplesTrain);
    dataTrain = allData(:,ind);

    if fMMcase == 1
        save MM1_TrainData dataTrain;
        save MM1_allData allData;
    elseif fMMcase == 2
        save MM2_TrainData dataTrain;
        save MM2_allData allData;
    elseif fMMcase == 3
        save MM3_TrainData dataTrain;
        save MM3_allData allData;
    end
else
    if fMMcase == 1
        load MM1_TrainData dataTrain;
        load MM1_allData allData;
    elseif fMMcase == 2
        load MM2_TrainData dataTrain;
        load MM2_allData allData;
    elseif fMMcase == 3
        load MM3_TrainData dataTrain;
        load MM3_allData allData;
    end
end

%%%%%%%% TEST DATA PREPARATION %%%%%%%%%%
if fTestData
    %% multiple ICs: 5 per Spec
    nICs = 5^nSpecs;
    y_init = rand(2,nICs);
    if fMMcase == 1
        y_init(1,:) = 1e+3 + 1e+2*y_init(1,:);   %% MM1 case
    elseif fMMcase == 2
        y_init(2,:) = 0.5*e0+y_init(2,:);        %% MM2 case
    elseif fMMcase == 3
        y_init(1,:) = 1e+2 + 1e+1*y_init(1,:);   %% MM3 case: k2>k1b+k1f*s (so that 2nd is fast)
        y_init(2,:) = e0 - 0.1*e0*y_init(2,:);   %% MM3 case
    end
    allData = zeros(nSpecs+1,nICs*nSamplesPT);
    for i = 1:nICs
        parVec = [k1f k1b k2 e0 y_init(1,i) y_init(2,i)];
        yAll = getMMsolGRID(parVec,nSpecs,nSamplesPT,min(DomC,[],'all'));
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

    if fMMcase == 1
        save MM1_TestData allData;
    elseif fMMcase == 2
        save MM2_TestData allData;
    elseif fMMcase == 3
        save MM3_TestData allData;
    end
else
    if fMMcase == 1
        load MM1_TestData allData;
    elseif fMMcase == 2
        load MM2_TestData allData;
    elseif fMMcase == 3
        load MM3_TestData allData;
    end
end

return