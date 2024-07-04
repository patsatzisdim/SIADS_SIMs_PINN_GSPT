clc
clear

% flag for plots
fPlot = 2;       % 1 is for plotting the SIM
                 % 2 for training/testing data
                 % 3 for errors per point, PIML and all GSPT expressions (select which to show)
if fPlot == 3
    fSIMapp = 11;       % 1-11 for PIML, QSSAL, QSSAR, QSSARL, PEA, CSPL11, CSPL21, CSPR11, CSPR21, CSPRL11, CSPRL21
end
fTMDDper = 4;     % 2 or 4 for the TMDD periods P2 and P4 when the SIM arises

%% visualizations: 1, the SIM in Omega
if fPlot==1
    inSz = 3;               % number of variables
    % testing data (on SIM) are not good for visualization
    % we create a different, structured, set here
    ft = 1;                 % first index is time
    if fTMDDper == 2
        fidx = 2+ft;                      % fast variable index
        sidx = [1+ft; 3+ft];              % slow variables indices
    elseif fTMDDper == 4
        fidx = 1+ft;                      % fast variable index
        sidx = [2+ft; 3+ft];              % slow variables indices
    end
    % parameter sets
    kon = 0.091;
    koff = 0.001;
    kel = 0.0015;
    ksyn = 0.11;
    kdeg = 0.0089;
    kint = 0.003;
    % domains Omega
    if fTMDDper == 2
        DomC = [10 140; 1e-3 0.12; 10 33];             % domain Omega to visualize SIM of P2
    elseif fTMDDper == 4
        DomC = [5e-3 5e-2; 6 13; 6 27];          % domain Omega to visualize SIM of P4
    end
    % initial conditions (ICs)
    if fTMDDper == 2
        sv1_grid = linspace(50,150,25);   % same ICs for L but for uniform grid, R, RL start at equilibirum
    elseif fTMDDper == 4
        sv1_grid = linspace(50,150,25);
    end
    y_init = [sv1_grid; ones(1,25)*ksyn/kdeg; zeros(1,25)];
    % trajectory points
    nSamplesPT = 100;
    nICs = size(y_init,2);   
    allData = zeros(inSz+1,nICs*nSamplesPT);
    for i = 1:nICs
        parVec = [kon koff kel ksyn kdeg kint y_init(1,i) y_init(2,i) y_init(3,i)];
        yAll = getTMDDsolGRID(parVec,inSz,nSamplesPT,fTMDDper);
        allData(:,(i-1)*nSamplesPT+1:i*nSamplesPT) = yAll; 
    end    
    Xdata = allData(fidx,:);
    Ydata = allData(sidx,:);

    % form the SIM surface 
    % in the case, where each trajectory has the same number of points, use the unstructured quadrilateral grid on the surface
    x_grid = reshape(Ydata(1,:),[nSamplesPT nICs]);
    y_grid = reshape(Ydata(2,:),[nSamplesPT nICs]);
    z_grid = reshape(Xdata(1,:),[nSamplesPT nICs]);
    % in the case where each trajecotory has different number of points, use delaunay triangulation to construct the grid on the surface 

    % show some trajectories approaching the SIM
    tend = 5000;
    tspan = [0 tend];
    opts = odeset('RelTol',1e-10,'AbsTol',1e-10);
    if (fTMDDper == 2) || (fTMDDper == 4)
        y_init1 = [120; ksyn/kdeg; 0];
        y_init2 = [80; ksyn/kdeg; 0];
    end
    sol1 = ode15s(@(t,y) TMDDode(t,y,kon,koff,kel,ksyn,kdeg,kint),tspan,y_init1,opts);
    sol2 = ode15s(@(t,y) TMDDode(t,y,kon,koff,kel,ksyn,kdeg,kint),tspan,y_init2,opts);
    if fTMDDper == 2
        offsetZsol = 1e-3; % offset on z-axis of trajectories to be plotted above the surface
    elseif fTMDDper == 4
        offsetZsol = 3e-4;
    end

    % plotting
    figure(fTMDDper);
    set(gcf,'renderer','Painters');
    ax = axes('OuterPosition',[0 0 1 1],'PositionConstraint','outerposition');
    surf(ax,x_grid,y_grid,z_grid,z_grid,'EdgeColor','interp','FaceColor','interp'); hold on;
    % manually find the point to plot triangle markers fo dirrection
    if fTMDDper == 2
        scatter3(ax,sol1.y(sidx(1)-ft,254),sol1.y(sidx(2)-ft,254),sol1.y(fidx-ft,254)+offsetZsol,100,'rv','filled'); hold on;
        scatter3(ax,sol2.y(sidx(1)-ft,268),sol2.y(sidx(2)-ft,268),sol2.y(fidx-ft,268)+offsetZsol,100,'rv','filled'); hold on;
    elseif fTMDDper == 4
        scatter3(ax,sol1.y(sidx(1)-ft,945),sol1.y(sidx(2)-ft,945),sol1.y(fidx-ft,945)+offsetZsol,100,'rv','filled'); hold on;
        scatter3(ax,sol2.y(sidx(1)-ft,926),sol2.y(sidx(2)-ft,926),sol2.y(fidx-ft,926)+offsetZsol,100,'rv','filled'); hold on;
    end
    plot3(ax,sol1.y(sidx(1)-ft,:),sol1.y(sidx(2)-ft,:),sol1.y(fidx-ft,:)+offsetZsol,'r-','LineWidth',2); hold on;
    plot3(ax,sol2.y(sidx(1)-ft,:),sol2.y(sidx(2)-ft,:),sol2.y(fidx-ft,:)+offsetZsol,'r-','LineWidth',2); 
    hold off;
    ax.XScale = 'linear';
    ax.YScale = 'linear';
    ax.ZScale = 'linear';
    if fTMDDper == 2      
        offsetx = 0;
        offsety = 0;
        offsetz = 0.02;
        ax.XTick = linspace(40,120,3);
        ax.YTick = linspace(10,30,3);
        ax.ZTick = linspace(0,0.1,3);
        view(55,25);
    elseif fTMDDper == 4  
        offsetx = 1;
        offsety = 1;
        offsetz = 0.01;
        ax.XTick = linspace(6,12,3);
        ax.YTick = linspace(8,24,3);
        ax.ZTick = linspace(0,0.06,3);
        view(35,25);
    end
    ax.XLim = [DomC(sidx(1)-ft,1)-offsetx DomC(sidx(1)-ft,2)];
    ax.YLim = [DomC(sidx(2)-ft,1) DomC(sidx(2)-ft,2)+offsety];
    ax.ZLim = [0 DomC(fidx-ft,2)+offsetz];
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';
    ax.ZMinorTick = 'on';
    
    ax.FontName = 'times';
    ax.FontSize = 20;
    ax.LabelFontSizeMultiplier = 24/20;
    ax.TickLabelInterpreter = 'latex';
    if fTMDDper == 2
        ax.XLabel.String = '$L$';
        ax.YLabel.String = '$RL$';
        ax.ZLabel.String = '$R$';
    elseif fTMDDper == 4
        ax.XLabel.String = '$R$';
        ax.YLabel.String = '$RL$';
        ax.ZLabel.String = '$L$';
    end
    ax.XLabel.Interpreter = 'latex';
    ax.XLabel.FontSize = 24;
    ax.YLabel.Interpreter = 'latex';
    ax.YLabel.FontSize = 24;
    ax.ZLabel.Interpreter = 'latex';
    ax.ZLabel.FontSize = 24;
    ax.ZLabel.Rotation = 0;

    prePosition = ax.Position;
    if fTMDDper == 2
        ax.XLabel.Position(1) = 60;
        ax.XLabel.Position(2) = 9;  
        ax.XLabel.Position(3) = -0.035;
        ax.YLabel.Position(1) = 140;
        ax.YLabel.Position(2) = 27;
        ax.YLabel.Position(3) = -0.03;
        ax.ZLabel.Position(1)= -6;
        ax.ZLabel.Position(2)= 6;
        ax.ZLabel.Position(3) = 0.065;
    elseif fTMDDper == 4
        ax.XLabel.Position(1) = 9;
        ax.XLabel.Position(2) = 9;  
        ax.XLabel.Position(3) = -0.017;
        ax.YLabel.Position(1)=13;
        ax.YLabel.Position(2)=15;
        ax.YLabel.Position(3)=-0.014;
        ax.ZLabel.Position(1) = 3.56;
        ax.ZLabel.Position(2) = 3.3;
        ax.ZLabel.Position(3) = 0.04;
        
    end
    ax.Position  = prePosition;
    set(gcf,'PaperPositionMode',"auto");
    return
    % huge, but good quality figure
    % print -depsc -tiff -r300 -painters ../../Figs_GSPT/TMDDP2_SIM.eps
    % much smaller for fast compile
    % print -dpng -r300 -painters ../../Figs_GSPT/paper/TMDDP2_SIM_l.png

%% Visualization of training/testing data
elseif fPlot == 2
    inSz = 3;               % number of variables
    ft = 1;                 % first index is time
    if fTMDDper == 2
        load TMDDP2_TestData allData;
        load TMDDP2_TrainData dataTrain;
        fidx = 2+ft;                      % fast variable index
        sidx = [1+ft; 3+ft];              % slow variables indices
    elseif fTMDDper == 4
        load TMDDP4_TestData.mat allData;
        load TMDDP4_TrainData dataTrain;
        fidx = 1+ft;                      % fast variable index
        sidx = [2+ft; 3+ft];              % slow variables indices
    end
    XTestData = allData(fidx,:);
    YTestData = allData(sidx,:);
    XTrainData = dataTrain(fidx,:);
    YTrainData = dataTrain(sidx,:);

    % domains Omega
    if fTMDDper == 2
        DomC = [10 140; 1e-3 0.12; 10 33];       % domain Omega to visualize SIM of P2
        offsetZTrain = 2e-3;                     % minor offset for better visualization     
    elseif fTMDDper == 4
        DomC = [5e-3 5e-2; 6 13; 6 27];          % domain Omega to visualize SIM of P4
        offsetZTrain = 5e-4;                     % minor offset for better visualization  
    end

    % plotting
    figure(fTMDDper);
    set(gcf,'renderer','Painters');
    ax = axes('OuterPosition',[0 0 1 1],'PositionConstraint','outerposition');
    scatter3(ax,YTestData(1,:),YTestData(2,:),XTestData,'r.','DisplayName','Test set'); hold on;
    scatter3(ax,YTrainData(1,:),YTrainData(2,:),XTrainData+offsetZTrain,24,'bo','LineWidth',0.8,'DisplayName','Train set'); hold off;
    ax.XScale = 'linear';
    ax.YScale = 'linear';
    ax.ZScale = 'linear';
    if fTMDDper == 2      
        offsetx = 0;
        offsety = 0;
        offsetz = 0.02;
        ax.XTick = linspace(40,120,3);
        ax.YTick = linspace(10,30,3);
        ax.ZTick = linspace(0,0.1,3);
        view(55,25);
    elseif fTMDDper == 4  
        offsetx = 1;
        offsety = 1;
        offsetz = 0.01;
        ax.XTick = linspace(6,12,3);
        ax.YTick = linspace(8,24,3);
        ax.ZTick = linspace(0,0.06,3);
        view(35,25);
    end
    ax.XLim = [DomC(sidx(1)-ft,1)-offsetx DomC(sidx(1)-ft,2)];
    ax.YLim = [DomC(sidx(2)-ft,1) DomC(sidx(2)-ft,2)+offsety];
    ax.ZLim = [0 DomC(fidx-ft,2)+offsetz];
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
    if fTMDDper == 2
        ax.XLabel.String = '$L$';
        ax.YLabel.String = '$RL$';
        ax.ZLabel.String = '$R$';
        lgd.Location = 'none';   % first place it, then move it.
        lgd.Position(1) = 0.61;
        lgd.Position(2) = 0.3;  
    elseif fTMDDper == 4
        ax.XLabel.String = '$R$';
        ax.YLabel.String = '$RL$';
        ax.ZLabel.String = '$L$';
        lgd.Location = 'none';   % first place it, then move it.
        lgd.Position(1) = 0.2;
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
    if fTMDDper == 2
        ax.XLabel.Position(1) = 60;
        ax.XLabel.Position(2) = 9;  
        ax.XLabel.Position(3) = -0.035;
        ax.YLabel.Position(1) = 140;
        ax.YLabel.Position(2) = 27;
        ax.YLabel.Position(3) = -0.03;
        ax.ZLabel.Position(1)= -6;
        ax.ZLabel.Position(2)= 6;
        ax.ZLabel.Position(3) = 0.065;
    elseif fTMDDper == 4
        ax.XLabel.Position(1) = 9;
        ax.XLabel.Position(2) = 9;  
        ax.XLabel.Position(3) = -0.017;
        ax.YLabel.Position(1)=13;
        ax.YLabel.Position(2)=15;
        ax.YLabel.Position(3)=-0.014;
        ax.ZLabel.Position(1) = 3.56;
        ax.ZLabel.Position(2) = 3.3;
        ax.ZLabel.Position(3) = 0.04;
    end
    ax.Position  = prePosition;
    set(gcf,'PaperPositionMode',"auto");

    return
    % huge, but good quality figure
    % print -depsc -tiff -r300 -painters ../../Figs_GSPT/TMDDP2_DataSets.eps
    % much smaller for fast compile
    % print -dpng -r300 -painters ../../Figs_GSPT/TMDDP2_DataSets_l.png

elseif fPlot == 3
    inSz = 3;               % number of variables
    ft = 1;                 % first index is time
    if fTMDDper == 2
        load TMDDP2_TestData allData;
        fidx = 2+ft;                      % fast variable index
        sidx = [1+ft; 3+ft];              % slow variables indices
    elseif fTMDDper == 4
        load TMDDP4_TestData allData;
        fidx = 1+ft;                      % fast variable index
        sidx = [2+ft; 3+ft];              % slow variables indices
    end
    Xdata = allData(fidx,:);
    Ydata = allData(sidx,:);
    
    % parameter sets
    kon = 0.091;
    koff = 0.001;
    kel = 0.0015;
    ksyn = 0.11;
    kdeg = 0.0089;
    kint = 0.003;

    % domains Omega
    if fTMDDper == 2
        DomC = [10 140; 1e-3 0.12; 10 33];       % domain Omega to visualize SIM of P2 
    elseif fTMDDper == 4
        DomC = [5e-3 5e-2; 6 13; 6 27];          % domain Omega to visualize SIM of P4  
    end

    % load trained parameters
    netDim = [3 20 1 3 20 2 2 20 1];
    if fTMDDper == 2
        load learned_P2 bestLearned;
    elseif fTMDDper == 4
        load learned_P4 bestLearned;
    end

    % find errors of explicit SIM expressions
    % PIML
    Xz = Xtrans(allData(2:4,:),bestLearned,netDim);
    Yz = Ytrans(allData(2:4,:),bestLearned,netDim);
    PIML_SIM = forwardNN3(Yz,bestLearned,netDim);
    AErr_PIML = abs(PIML_SIM-Xz);
    % GSPT expressions
    pars = [kon koff kel ksyn kdeg kint];
    fImpNR = true;  
    [QSSAL_SIM, QSSAR_SIM,  QSSARL_SIM, PEA1_SIM, CSPL11_SIM, CSPL21_SIM, ...
        CSPR11_SIM, CSPR21_SIM, CSPRL11_SIM, CSPRL21_SIM] = TMDD_knownSIMs(allData(2:4,:),pars,fidx-1,fImpNR);

    AErr_QSSAL = abs(QSSAL_SIM-allData(fidx,:));
    AErr_QSSAR = abs(QSSAR_SIM-allData(fidx,:));
    AErr_QSSARL = abs(QSSARL_SIM-allData(fidx,:));
    AErr_PEA = abs(PEA1_SIM-allData(fidx,:));
    AErr_CSPL11 = abs(CSPL11_SIM-allData(fidx,:));
    AErr_CSPL21 = abs(CSPL21_SIM-allData(fidx,:));
    AErr_CSPR11 = abs(CSPR11_SIM-allData(fidx,:));
    AErr_CSPR21 = abs(CSPR21_SIM-allData(fidx,:));
    AErr_CSPRL11 = abs(CSPRL11_SIM-allData(fidx,:));
    AErr_CSPRL21 = abs(CSPRL21_SIM-allData(fidx,:));
    % make exact zeros the next higher value 
    AErr_PIML = findZTR(AErr_PIML);
    AErr_QSSAL = findZTR(AErr_QSSAL);
    AErr_QSSAR = findZTR(AErr_QSSAR);
    AErr_QSSARL = findZTR(AErr_QSSARL);
    AErr_PEA = findZTR(AErr_PEA);
    AErr_CSPL11 = findZTR(AErr_CSPL11);
    AErr_CSPL21 = findZTR(AErr_CSPL21);
    AErr_CSPR11 = findZTR(AErr_CSPR11);
    AErr_CSPR21 = findZTR(AErr_CSPR21);
    AErr_CSPRL11 = findZTR(AErr_CSPRL11);
    AErr_CSPRL21 = findZTR(AErr_CSPRL21);

    % max and min for colorbar
    minAE = min([AErr_PIML AErr_QSSAL AErr_QSSAR AErr_QSSARL AErr_PEA AErr_CSPL11 AErr_CSPL21 AErr_CSPR11 AErr_CSPR21 AErr_CSPRL11 AErr_CSPRL21]);
    maxAE = max([AErr_PIML AErr_QSSAL AErr_QSSAR AErr_QSSARL AErr_PEA AErr_CSPL11 AErr_CSPL21 AErr_CSPR11 AErr_CSPR21 AErr_CSPRL11 AErr_CSPRL21]);
    minAE = ceil(log10(minAE));
    maxAE = floor(log10(maxAE));

    % ploting
    figure(fSIMapp);
    set(gcf,'renderer','Painters');
    ax = axes('OuterPosition',[0 0 1 1],'PositionConstraint','outerposition');
    if fSIMapp == 1
        scatter3(ax,Ydata(1,:),Ydata(2,:),Xdata(1,:),50,log10(AErr_PIML),'.');
    elseif fSIMapp == 2
        scatter3(ax,Ydata(1,:),Ydata(2,:),Xdata(1,:),50,log10(AErr_QSSAL),'.');
    elseif fSIMapp == 3
        scatter3(ax,Ydata(1,:),Ydata(2,:),Xdata(1,:),50,log10(AErr_QSSAR),'.');
    elseif fSIMapp == 4
        scatter3(ax,Ydata(1,:),Ydata(2,:),Xdata(1,:),50,log10(AErr_QSSARL),'.');
    elseif fSIMapp == 5
        scatter3(ax,Ydata(1,:),Ydata(2,:),Xdata(1,:),50,log10(AErr_PEA),'.');
    elseif fSIMapp == 6
        scatter3(ax,Ydata(1,:),Ydata(2,:),Xdata(1,:),50,log10(AErr_CSPL11),'.');
    elseif fSIMapp == 7
        scatter3(ax,Ydata(1,:),Ydata(2,:),Xdata(1,:),50,log10(AErr_CSPL21),'.');
    elseif fSIMapp == 8
        scatter3(ax,Ydata(1,:),Ydata(2,:),Xdata(1,:),50,log10(AErr_CSPR11),'.');
    elseif fSIMapp == 9
        scatter3(ax,Ydata(1,:),Ydata(2,:),Xdata(1,:),50,log10(AErr_CSPR21),'.');
    elseif fSIMapp == 10
        scatter3(ax,Ydata(1,:),Ydata(2,:),Xdata(1,:),50,log10(AErr_CSPRL11),'.');
    elseif fSIMapp == 11
        scatter3(ax,Ydata(1,:),Ydata(2,:),Xdata(1,:),50,log10(AErr_CSPRL21),'.');
    end
    ax.XScale = 'linear';
    ax.YScale = 'linear';
    ax.ZScale = 'linear';
    if fTMDDper == 2      
        offsetx = 0;
        offsety = 0;
        offsetz = 0.02;
        ax.XTick = linspace(40,120,3);
        ax.YTick = linspace(10,30,3);
        ax.ZTick = linspace(0,0.1,3);
        view(55,25);
    elseif fTMDDper == 4  
        offsetx = 1;
        offsety = 1;
        offsetz = 0.01;
        ax.XTick = linspace(6,12,3);
        ax.YTick = linspace(8,24,3);
        ax.ZTick = linspace(0,0.06,3);
        view(35,25);
    end
    ax.XLim = [DomC(sidx(1)-ft,1)-offsetx DomC(sidx(1)-ft,2)];
    ax.YLim = [DomC(sidx(2)-ft,1) DomC(sidx(2)-ft,2)+offsety];
    ax.ZLim = [0 DomC(fidx-ft,2)+offsetz];
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';
    ax.ZMinorTick = 'on';

    % colorbar
    c = colorbar;
    colormap(jet)
    caxis(ax,[minAE maxAE]);
    if fTMDDper == 2
        c.Ticks = -14:4:-2;
    elseif fTMDDper == 4
        caxis(ax,[minAE maxAE+1]);
        c.Ticks = -11:3:-2;
    end
    c.Label.Interpreter = 'latex';
    c.Label.String = 'log(ae$^{(i)}$)';
    c.Label.FontSize = 20;

    ax.FontName = 'times';
    ax.FontSize = 20;
    ax.LabelFontSizeMultiplier = 24/20;
    ax.TickLabelInterpreter = 'latex';
    if fTMDDper == 2
        ax.XLabel.String = '$L$';
        ax.YLabel.String = '$RL$';
        ax.ZLabel.String = '$R$'; 
    elseif fTMDDper == 4
        ax.XLabel.String = '$R$';
        ax.YLabel.String = '$RL$';
        ax.ZLabel.String = '$L$';
    end
    ax.XLabel.Interpreter = 'latex';
    ax.XLabel.FontSize = 24;
    ax.YLabel.Interpreter = 'latex';
    ax.YLabel.FontSize = 24;
    ax.ZLabel.Interpreter = 'latex';
    ax.ZLabel.FontSize = 24;
    ax.ZLabel.Rotation = 0;

    prePosition = ax.Position;
    if fTMDDper == 2
        ax.XLabel.Position(1) = 60;
        ax.XLabel.Position(2) = 9;  
        ax.XLabel.Position(3) = -0.035;
        ax.YLabel.Position(1) = 140;
        ax.YLabel.Position(2) = 27;
        ax.YLabel.Position(3) = -0.03;
        ax.ZLabel.Position(1)= -8;
        ax.ZLabel.Position(2)= 5;
        ax.ZLabel.Position(3) = 0.065;
    elseif fTMDDper == 4
        ax.XLabel.Position(1) = 8.5;
        ax.XLabel.Position(2) = 9;  
        ax.XLabel.Position(3) = -0.017;
        ax.YLabel.Position(1) = 13;
        ax.YLabel.Position(2) = 15;
        ax.YLabel.Position(3) = -0.014;
        ax.ZLabel.Position(1) = 3.56;
        ax.ZLabel.Position(2) = 1.8;
        ax.ZLabel.Position(3) = 0.04;
    end
    ax.Position  = prePosition;
    set(gcf,'PaperPositionMode',"auto");

    return
    % huge, but good quality figure
    % print -depsc -tiff -r300 -painters ../../Figs_GSPT/TMDDP2_AE_PIML.eps
    % much smaller for fast compile
    % print -dpng -r300 -painters ../../Figs_GSPT/light(png)_versions/TMDDP2_AE_PIML_l.png

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

% function that find zeros in array and replaces the value with the next higher value in array
function arrayX = findZTR(arrayX)
    zerIdx = find(arrayX==0);
    minVal = mink(arrayX,numel(zerIdx)+1);
    arrayX(zerIdx) = minVal(end);
end