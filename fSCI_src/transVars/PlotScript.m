%% Script including all the visualizations
clc
clf
clear

% flag for plots

fPlot = 1;       % 1 is for plotting the SIM
                 % 2 for training/testing data
                 % 3 for errors per point, PIML and all GSPT expressions (select which to show)
if fPlot == 3
    fSIMapp = 5;       % 1-11 for PIML, QSSA, PEA, CSP11, CSP21
end
fProj = 2;       % 1 or 2 for showing the projection to the 1st or 2nd fast variable

%% visualizations: 1, the SIM in Omega
if fPlot==1
    inSz = 4;               % number of variables
    % testing data (on SIM) are not good for visualization
    % we create a different, structured, set here
    ft = 1;                 % first index is time
    fidx = [2+ft; 4+ft];              % fast variable index
    sidx = [1+ft; 3+ft];              % slow variables indices

    % parameter sets
    KM1 = 23;
    k2 = 42/60;
    k1b = 4*k2;
    k1f = 5*k2/KM1;
    KM2 = 25;
    k4 = 25/60;
    k3b = 4*k4;
    k3f = 5*k4/KM2;
    e0 = 50;
    % domain Omega
    DomC = [1e-5 80; 1e-5 30; 1e-3 80; 1e-3 30];           
    % initial conditions (ICs)
    sv1_grid = linspace(50,151,5);   % ICs for s1 for uniform grid, more extended thatn test Data to show better boundaries
    sv2_grid = linspace(50,151,5);   % ICs for s2 for uniform grid
    [sv1, sv2] = meshgrid(sv1_grid,sv2_grid);
    y_init = [reshape(sv2,1,[]); zeros(1,25) ; reshape(sv1,1,[]); zeros(1,25)];
    % trajectory points
    nSamplesPT = 100;
    nICs = size(y_init,2);   
    allData = zeros(inSz+1,nICs*nSamplesPT);
    for i = 1:nICs
        parVec = [k1f k1b k2 k3f k3b k4 e0 y_init(1,i) y_init(2,i) y_init(3,i) y_init(4,i)];
        yAll = getINHsolGRID(parVec,inSz,nSamplesPT,min(DomC,[],'all'));
        allData(:,(i-1)*nSamplesPT+1:i*nSamplesPT) = yAll; 
    end    
    Xdata = allData(fidx,:);
    Ydata = allData(sidx,:);

    % form the SIM surface 
    % in the case, where each trajectory has the same number of points, use the unstructured quadrilateral grid on the surface
    x_grid = reshape(Ydata(1,:),[nSamplesPT nICs]);
    y_grid = reshape(Ydata(2,:),[nSamplesPT nICs]);
    z_grid = reshape(Xdata(fProj,:),[nSamplesPT nICs]);
    % in the case where each trajecotory has different number of points, use delaunay triangulation to construct the grid on the surface 

    % show some trajectories approaching the SIM
    tend = 5000;
    tspan = [0 tend];
    opts = odeset('RelTol',1e-10,'AbsTol',1e-10);
    if (fProj == 1) || (fProj == 2)
        y_init1 = [40; 1e-3; 35; 1e-1];
        y_init2 = [2; 1e-4; 15; 1e-2];
    end
    sol1 = ode15s(@(t,y) INHode(t,y,k1f,k1b,k2,k3f,k3b,k4,e0),tspan,y_init1,opts);
    sol2 = ode15s(@(t,y) INHode(t,y,k1f,k1b,k2,k3f,k3b,k4,e0),tspan,y_init2,opts);
    if fProj == 1
        offsetZsol = 1e-6; % offset on z-axis of trajectories to be plotted above the surface
    elseif fProj == 2
        offsetZsol = 1e-3;
    end

    % plotting
    figure(fProj);
    set(gcf,'renderer','Painters');
    ax = axes('OuterPosition',[0 0 1 1],'PositionConstraint','outerposition');
    surf(ax,x_grid,y_grid,z_grid,z_grid,'EdgeColor','interp','FaceColor','interp'); hold on;
    scatter3(ax,sol1.y(sidx(1)-ft,1),sol1.y(sidx(2)-ft,1),sol1.y(fidx(fProj)-ft,1)+offsetZsol,100,'rs','filled'); hold on;
    scatter3(ax,sol2.y(sidx(1)-ft,1),sol2.y(sidx(2)-ft,1),sol2.y(fidx(fProj)-ft,1)+offsetZsol,100,'rs','filled'); hold on;
    plot3(ax,sol1.y(sidx(1)-ft,:),sol1.y(sidx(2)-ft,:),sol1.y(fidx(fProj)-ft,:)+offsetZsol,'r-','LineWidth',2); hold on;
    plot3(ax,sol2.y(sidx(1)-ft,:),sol2.y(sidx(2)-ft,:),sol2.y(fidx(fProj)-ft,:)+offsetZsol,'r-','LineWidth',2); 
    hold off;
    ax.XScale = 'log';
    ax.YScale = 'log';
    ax.ZScale = 'log';
    offsetx = 1e-6;
    offsety = 0;
    ax.XTick = logspace(-5,1,3);
    ax.YTick = logspace(-3,1,3);
    if fProj == 1            
        offsetz = 1e-6;
        ax.ZTick = logspace(-5,1,3);
        view(-35,25);
    elseif fProj == 2  
        offsetz = 0;
        ax.ZTick = logspace(-3,1,3);
        view(45,25);
    end
    ax.XLim = [DomC(sidx(1)-ft,1)-offsetx DomC(sidx(1)-ft,2)];
    ax.YLim = [DomC(sidx(2)-ft,1)+offsety DomC(sidx(2)-ft,2)+offsety];
    ax.ZLim = [DomC(fidx(fProj)-ft,1)-offsetz DomC(fidx(fProj)-ft,2)];
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';
    ax.ZMinorTick = 'on';
    
    ax.FontName = 'times';
    ax.FontSize = 20;
    ax.LabelFontSizeMultiplier = 24/20;
    ax.TickLabelInterpreter = 'latex';
    if fProj == 1
        ax.XLabel.String = '$\bar{s}_1$';
        ax.YLabel.String = '$\bar{s}_2$';
        ax.ZLabel.String = '$c_1$';
    elseif fProj == 2
        ax.XLabel.String = '$\bar{s}_1$';
        ax.YLabel.String = '$\bar{s}_2$';
        ax.ZLabel.String = '$c_2$';
    end
    ax.XLabel.Interpreter = 'latex';
    ax.XLabel.FontSize = 24;
    ax.YLabel.Interpreter = 'latex';
    ax.YLabel.FontSize = 24;
    ax.ZLabel.Interpreter = 'latex';
    ax.ZLabel.FontSize = 24;
    ax.ZLabel.Rotation = 0;

    prePosition = ax.Position;
    if fProj == 1
        ax.XLabel.Position(1) = 0.008;
        ax.XLabel.Position(2) = 2e-3; 
        ax.XLabel.Position(3) = 4e-7;
        ax.YLabel.Position(1) = 4e-6;
        ax.YLabel.Position(2) = 0.4;
        ax.YLabel.Position(3) = 2e-7;
        ax.ZLabel.Position(1) = 5e-7;
        ax.ZLabel.Position(2) = 400;
        ax.ZLabel.Position(3) = 0.15;
    elseif fProj == 2
        ax.XLabel.Position(1) = 0.01;
        ax.XLabel.Position(2) = 1e-3;  
        ax.XLabel.Position(3) = 1e-4;
        ax.YLabel.Position(1) = 30;
        ax.YLabel.Position(2) = 0.2;
        ax.YLabel.Position(3) = 1e-4;
        ax.ZLabel.Position(1) = 6e-7;
        ax.ZLabel.Position(2) = 1.7e-4;
        ax.ZLabel.Position(3) = 0.6;
        
    end
    ax.Position  = prePosition;
    set(gcf,'PaperPositionMode',"auto");
    return
    % huge, but good quality figure
    % print -depsc -tiff -r300 -painters ../../../Figs_GSPT/paper/CompInhTr_SIM1.eps
    % much smaller for fast compile
    % print -dpng -r300 -painters ../../../Figs_GSPT/paper/CompInhTr_SIM1_l.png

%% Visualization of training/testing data
elseif fPlot == 2
    inSz = 4;               % number of variables
    ft = 1;                 % first index is time
    load CompInh_TestData allData;
    load CompInh_TrainData dataTrain;
    fidx = [2+ft; 4+ft];              % fast variable index
    sidx = [1+ft; 3+ft];              % slow variables indices
    XTestData = allData(fidx,:);
    YTestData = allData(sidx,:);
    XTrainData = dataTrain(fidx,:);
    YTrainData = dataTrain(sidx,:);
    % domain Omega
    DomC = [1e-5 80; 1e-5 30; 1e-3 80; 1e-3 30];  
    % minor offset for better visualization 
    if fProj == 1
        offsetZTrain = 0;                         
    elseif fProj == 2
        offsetZTrain = 0;                     
    end

    % plotting
    figure(fProj);
    set(gcf,'renderer','Painters');
    ax = axes('OuterPosition',[0 0 1 1],'PositionConstraint','outerposition');
    scatter3(ax,YTestData(1,:),YTestData(2,:),XTestData(fProj,:),'r.','DisplayName','Test set'); hold on;
    scatter3(ax,YTrainData(1,:),YTrainData(2,:),XTrainData(fProj,:)+offsetZTrain,24,'bo','LineWidth',0.8,'DisplayName','Train set'); hold off;
    ax.XScale = 'log';
    ax.YScale = 'log';
    ax.ZScale = 'log';
    offsetx = 1e-6;
    offsety = 0;
    ax.XTick = logspace(-5,1,3);
    ax.YTick = logspace(-3,1,3);
    if fProj == 1            
        offsetz = 1e-6;
        ax.ZTick = logspace(-5,1,3);
        view(-35,25);
    elseif fProj == 2  
        offsetz = 0;
        ax.ZTick = logspace(-3,1,3);
        view(45,25);
    end
    ax.XLim = [DomC(sidx(1)-ft,1)-offsetx DomC(sidx(1)-ft,2)];
    ax.YLim = [DomC(sidx(2)-ft,1)+offsety DomC(sidx(2)-ft,2)+offsety];
    ax.ZLim = [DomC(fidx(fProj)-ft,1)-offsetz DomC(fidx(fProj)-ft,2)];
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';
    ax.ZMinorTick = 'on';

    lgd = legend;
    lgd.FontName = 'times';
    lgd.FontSize = 20;
    lgd.Interpreter = 'latex';
    lgd.Location = 'southeast';

    ax.FontName = 'times';
    ax.FontSize = 20;
    ax.LabelFontSizeMultiplier = 24/20;
    ax.TickLabelInterpreter = 'latex';
    if fProj == 1
        ax.XLabel.String = '$\bar{s}_1$';
        ax.YLabel.String = '$\bar{s}_2$';
        ax.ZLabel.String = '$c_1$';
        lgd.Location = 'none';
        lgd.Position(1) = 0.6;
        lgd.Position(2) = 0.3;
    elseif fProj == 2
        ax.XLabel.String = '$\bar{s}_1$';
        ax.YLabel.String = '$\bar{s}_2$';
        ax.ZLabel.String = '$c_2$';
        lgd.Location = 'none';
        lgd.Position(1) = 0.6;
        lgd.Position(2) = 0.3;
    end
    ax.XLabel.Interpreter = 'latex';
    ax.XLabel.FontSize = 24;
    ax.YLabel.Interpreter = 'latex';
    ax.YLabel.FontSize = 24;
    ax.ZLabel.Interpreter = 'latex';
    ax.ZLabel.FontSize = 24;
    ax.ZLabel.Rotation = 0;

    prePosition = ax.Position;
    if fProj == 1
        ax.XLabel.Position(1) = 0.008;
        ax.XLabel.Position(2) = 2e-3; 
        ax.XLabel.Position(3) = 4e-7;
        ax.YLabel.Position(1) = 4e-6;
        ax.YLabel.Position(2) = 0.4;
        ax.YLabel.Position(3) = 2e-7;
        ax.ZLabel.Position(1) = 5e-7;
        ax.ZLabel.Position(2) = 400;
        ax.ZLabel.Position(3) = 0.15;
    elseif fProj == 2
        ax.XLabel.Position(1) = 0.01;
        ax.XLabel.Position(2) = 1e-3;  
        ax.XLabel.Position(3) = 1e-4;
        ax.YLabel.Position(1) = 30;
        ax.YLabel.Position(2) = 0.2;
        ax.YLabel.Position(3) = 1e-4;
        ax.ZLabel.Position(1) = 6e-7;
        ax.ZLabel.Position(2) = 1.7e-4;
        ax.ZLabel.Position(3) = 0.6;
        
    end
    ax.Position  = prePosition;
    set(gcf,'PaperPositionMode',"auto");

    return
    % huge, but good quality figure
    % print -depsc -tiff -r300 -painters ../../../Figs_GSPT/CompInhTr_DataSets1.eps
    % much smaller for fast compile
    % print -dpng -r300 -painters ../../../Figs_GSPT/light(png)_versions/CompInhTr_DataSets1_l.png

elseif fPlot == 3
    inSz = 4;               % number of variables
    ft = 1;                 % first index is time
    load CompInh_TestData allData;
    fidx = [2+ft; 4+ft];              % fast variable index
    sidx = [1+ft; 3+ft];              % slow variables indices
    Xdata = allData(fidx,:);
    Ydata = allData(sidx,:);
    
    % parameter sets
    KM1 = 23;
    k2 = 42/60;
    k1b = 4*k2;
    k1f = 5*k2/KM1;
    KM2 = 25;
    k4 = 25/60;
    k3b = 4*k4;
    k3f = 5*k4/KM2;
    e0 = 50;

    % domain Omega
    DomC = [1e-5 80; 1e-5 30; 1e-3 80; 1e-3 30];

    % load trained parameters
    netDim = [4 20 2 4 20 2 2 20 2];
    load PI_learned bestLearned;

    % find errors of explicit SIM expressions
    % PIML
    Xz = Xtrans(allData(2:5,:),bestLearned,netDim);
    Yz = Ytrans(allData(2:5,:),bestLearned,netDim);
    PIML_SIM = forwardNN3(Yz,bestLearned,netDim);
    AErr_PIML = abs(PIML_SIM-Xz);
    % GSPT expressions
    pars = [k1f k1b k2 k3f k3b k4 e0];
    fImpNR = true;  
    %[QSSAc1c2_SIM, PEA13c1c2_SIM, CSP11c1c2_SIM, CSP21c1c2_SIM] = CompInh_knownSIMs(allData(2:5,:),pars,fidx-1,fImpNR);
    %%%%%% precomputed for figs
    load QSSA_SIM QSSAc1c2_SIM;
    load PEA_SIM PEA13c1c2_SIM;
    load CSP11_SIM CSP11c1c2_SIM;
    load CSP21_SIM CSP21c1c2_SIM;
    AErr_QSSA = abs(QSSAc1c2_SIM-allData(fidx,:));
    AErr_PEA = abs(PEA13c1c2_SIM-allData(fidx,:));
    AErr_CSP11 = abs(CSP11c1c2_SIM-allData(fidx,:));
    AErr_CSP21 = abs(CSP21c1c2_SIM-allData(fidx,:));

    % make exact zeros the next higher value 
    AErr_PIML = findZTR(AErr_PIML);
    AErr_QSSA = findZTR(AErr_QSSA);
    AErr_PEA = findZTR(AErr_PEA);
    AErr_CSP11 = findZTR(AErr_CSP11);
    AErr_CSP21 = findZTR(AErr_CSP21);

    % max and min for colorbar
    minAE = min([AErr_PIML AErr_QSSA AErr_PEA AErr_CSP11 AErr_CSP21],[],'all');
    maxAE = max([AErr_PIML AErr_QSSA AErr_PEA AErr_CSP11 AErr_CSP21],[],'all');
    minAE = ceil(log10(minAE));
    maxAE = ceil(log10(maxAE));

    % ploting
    figure(fSIMapp);
    set(gcf,'renderer','Painters');
    ax = axes('OuterPosition',[0 0 1 1],'PositionConstraint','outerposition');
    if fSIMapp == 1
        scatter3(ax,Ydata(1,:),Ydata(2,:),Xdata(fProj,:),50,log10(AErr_PIML(fProj,:)),'.');
    elseif fSIMapp == 2
        scatter3(ax,Ydata(1,:),Ydata(2,:),Xdata(fProj,:),50,log10(AErr_QSSA(fProj,:)),'.');
    elseif fSIMapp == 3
        scatter3(ax,Ydata(1,:),Ydata(2,:),Xdata(fProj,:),50,log10(AErr_PEA(fProj,:)),'.');
    elseif fSIMapp == 4
        scatter3(ax,Ydata(1,:),Ydata(2,:),Xdata(fProj,:),50,log10(AErr_CSP11(fProj,:)),'.');
    elseif fSIMapp == 5
        scatter3(ax,Ydata(1,:),Ydata(2,:),Xdata(fProj,:),50,log10(AErr_CSP21(fProj,:)),'.');
    end
    ax.XScale = 'log';
    ax.YScale = 'log';
    ax.ZScale = 'log';
    offsetx = 1e-6;
    offsety = 0;
    ax.XTick = logspace(-5,1,3);
    ax.YTick = logspace(-3,1,3);
    if fProj == 1            
        offsetz = 1e-6;
        ax.ZTick = logspace(-5,1,3);
        view(-35,25);
    elseif fProj == 2  
        offsetz = 0;
        ax.ZTick = logspace(-3,1,3);
        view(45,25);
    end
    ax.XLim = [DomC(sidx(1)-ft,1)-offsetx DomC(sidx(1)-ft,2)];
    ax.YLim = [DomC(sidx(2)-ft,1)+offsety DomC(sidx(2)-ft,2)+offsety];
    ax.ZLim = [DomC(fidx(fProj)-ft,1)-offsetz DomC(fidx(fProj)-ft,2)];
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';
    ax.ZMinorTick = 'on';

    % colorbar
    c = colorbar;
    colormap(jet)
    caxis(ax,[minAE maxAE]);
    c.Ticks = -9:3:0;
    c.Label.Interpreter = 'latex';
    c.Label.String = 'log(ae$^{(i)}$)';
    c.Label.FontSize = 20;

    ax.FontName = 'times';
    ax.FontSize = 20;
    ax.LabelFontSizeMultiplier = 24/20;
    ax.TickLabelInterpreter = 'latex';
    if fProj == 1
        ax.XLabel.String = '$\bar{s}_1$';
        ax.YLabel.String = '$\bar{s}_2$';
        ax.ZLabel.String = '$c_1$';
        lgd.Location = 'none';
        lgd.Position(1) = 0.6;
        lgd.Position(2) = 0.3;
    elseif fProj == 2
        ax.XLabel.String = '$\bar{s}_1$';
        ax.YLabel.String = '$\bar{s}_2$';
        ax.ZLabel.String = '$c_2$';
        lgd.Location = 'none';
        lgd.Position(1) = 0.6;
        lgd.Position(2) = 0.3;
    end
    ax.XLabel.Interpreter = 'latex';
    ax.XLabel.FontSize = 24;
    ax.YLabel.Interpreter = 'latex';
    ax.YLabel.FontSize = 24;
    ax.ZLabel.Interpreter = 'latex';
    ax.ZLabel.FontSize = 24;
    ax.ZLabel.Rotation = 0;

    prePosition = ax.Position;
    if fProj == 1
        ax.XLabel.Position(1) = 0.01;
        ax.XLabel.Position(2) = 2e-3; 
        ax.XLabel.Position(3) = 4e-7;
        ax.YLabel.Position(1) = 4e-6;
        ax.YLabel.Position(2) = 0.5;
        ax.YLabel.Position(3) = 2e-7;
        ax.ZLabel.Position(1) = 5e-7;
        ax.ZLabel.Position(2) = 800;
        ax.ZLabel.Position(3) = 0.1;
    elseif fProj == 2
        ax.XLabel.Position(1) = 0.006;
        ax.XLabel.Position(2) = 1e-3;  
        ax.XLabel.Position(3) = 1e-4;
        ax.YLabel.Position(1) = 40;
        ax.YLabel.Position(2) = 0.3;
        ax.YLabel.Position(3) = 1e-4;
        ax.ZLabel.Position(1) = 2e-7;
        ax.ZLabel.Position(2) = 1.7e-4;
        ax.ZLabel.Position(3) = 0.5;
        
    end
    ax.Position  = prePosition;
    set(gcf,'PaperPositionMode',"auto");

    return
    % huge, but good quality figure
    % print -depsc -tiff -r300 -painters ../../../Figs_GSPT/CompInhTr_AE1_PIML.eps
    % much smaller for fast compile
    % print -dpng -r300 -painters ../../../Figs_GSPT/light(png)_versions/CompInhTr_AE1_PIML_l.png

end


%%%%%%%%%%%%%%% FUNCTION %%%%%%%%%%%%%%%%%%%%%%%%

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

% function that find zeros in array and replaces the value with the next higher value in array
function arrayX = findZTR(arrayX)
    zerIdx = find(arrayX==0);
    minVal = mink(arrayX,numel(zerIdx)+1);
    arrayX(zerIdx) = minVal(end);
end