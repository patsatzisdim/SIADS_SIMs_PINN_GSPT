clc
clear

% flag for plots
fPlot = 3;       % 1 is for plotting the SIM
                 % 2 for training/testing data
                 % 3 for errors per point, PIML and all GSPT expressions (select which to show)
if fPlot == 3
    fSIMapp = 9;       % 1-9 for PIML, sQSSA, rQSSA, PEA, CSPe, CSPc11, CSPc21, CSPs11, CSPs21
end
fMMcase = 3;     % 1, 2 or 3 case of MM considered

%% visualizations: 1, the SIM in Omega
if fPlot==1
    inSz = 2;               % number of variables
    % load testing data (on SIM) 
    ft = 1;                 % first index is time
    if fMMcase == 1
        load MM1_TestData allData;
        fidx = 1+ft;                      % fast variable index
        sidx = 2+ft;                      % slow variable index
    elseif fMMcase == 2
        load MM2_TestData allData;
        fidx = 2+ft;                      % fast variable index
        sidx = 1+ft;                      % slow variable index
    elseif fMMcase == 3
        load MM3_TestData allData;
        fidx = 1+ft;                      % fast variable index
        sidx = 2+ft;                      % slow variable index
    end
    Xdata = allData(fidx,:);
    Ydata = allData(sidx,:);

    % sort to visulize as line
    [~, idx] = sort(Xdata,'descend');
    Xdata = Xdata(1,idx);
    Ydata = Ydata(1,idx);

    % parameter sets
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
        DomC = [1e-5 1; 1e-3 1e+2];                % domain Omega to collect collocation points
    end
    
    % show some trajectories approaching the SIM
    tend = 1e+5;
    tspan = [0 tend];
    LB = DomC(:,1);   % lower bounds to cut trajectory
    opts = odeset('RelTol',1e-10,'AbsTol',1e-10,'Events',@(t,y) defineZero(t,y,LB));
    if fMMcase == 1
        y_init1 = [1e-2 ; 1e-1];
        y_init2 = [1e-2 ; 100];
    elseif fMMcase == 2
        y_init1 = [1e-1 ; 1e-2];
        y_init2 = [1e+1 ; 1e-2];
    elseif fMMcase == 3
        y_init1 = [1e-3 ; 1e-2];
        y_init2 = [1e-3 ; 10];
    end
    sol1 = ode15s(@(t,y) MMode(t,y,k1f,k1b,k2,e0),tspan,y_init1,opts);
    sol2 = ode15s(@(t,y) MMode(t,y,k1f,k1b,k2,e0),tspan,y_init2,opts);
 
    % plotting
    figure(fMMcase);
    set(gcf,'renderer','Painters');
    ax = axes('OuterPosition',[0 0 1 1],'PositionConstraint','outerposition');
    plot(ax,Ydata,Xdata,'b-','LineWidth',5); hold on;
    scatter(ax,sol1.y(sidx-ft,1),sol1.y(fidx-ft,1),100,'rs','filled'); hold on;
    plot(ax,sol1.y(sidx-ft,:),sol1.y(fidx-ft,:),'r-','LineWidth',2); hold on;
    scatter(ax,sol2.y(sidx-ft,1),sol2.y(fidx-ft,1),100,'rs','filled'); hold on;
    plot(ax,sol2.y(sidx-ft,:),sol2.y(fidx-ft,:),'r-','LineWidth',2); hold off;
    ax.XScale = 'log';
    ax.YScale = 'log';
    if fMMcase == 1      
        offsetx = 0;
        offsety = 1;
        ax.XTick = logspace(-3,3,4);
        ax.YTick = logspace(-4,0,3);
    elseif fMMcase == 2
        offsetx = 0;
        offsety = 1;
        ax.XTick = logspace(-3,1,3);
        ax.YTick = logspace(-4,0,3);
    elseif fMMcase == 3
        offsetx = 0;
        offsety = 1;
        ax.XTick = logspace(-3,1,3);
        ax.YTick = logspace(-4,0,3);
    end
    ax.XLim = [DomC(sidx-ft,1) DomC(sidx-ft,2)+offsetx];
    ax.YLim = [DomC(fidx-ft,1) DomC(fidx-ft,2)+offsety];
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';

    ax.FontName = 'times';
    ax.FontSize = 20;
    ax.LabelFontSizeMultiplier = 24/20;
    ax.TickLabelInterpreter = 'latex';
    if (fMMcase == 1) || (fMMcase == 3)
        ax.XLabel.String = '$s$';
        ax.YLabel.String = '$c$';
    elseif fMMcase == 2
        ax.XLabel.String = '$c$';
        ax.YLabel.String = '$s$';
    end
    ax.XLabel.Interpreter = 'latex';
    ax.XLabel.FontSize = 24;
    ax.YLabel.Interpreter = 'latex';
    ax.YLabel.FontSize = 24;
    ax.YLabel.Rotation = 0;

    prePosition = ax.Position;
    if fMMcase == 1
        ax.XLabel.Position(1) = 1.0;
        ax.XLabel.Position(2) = 3e-6;  
        ax.YLabel.Position(1) = 1.2e-4;
        ax.YLabel.Position(2) = 6e-4;
    elseif fMMcase == 2
        ax.XLabel.Position(1) = 1.0;
        ax.XLabel.Position(2) = 3e-6;  
        ax.YLabel.Position(1) = 1.9e-4;
        ax.YLabel.Position(2) = 6e-4;
    elseif fMMcase == 3
        ax.XLabel.Position(1) = 1.0;
        ax.XLabel.Position(2) = 3e-6;  
        ax.YLabel.Position(1) = 1.8e-4;
        ax.YLabel.Position(2) = 6e-4;
    end
    ax.Position  = prePosition;
    set(gcf,'PaperPositionMode',"auto");
    return
    % print -depsc -tiff -r300 -painters ../../Figs_paper2/MM1_SIM.eps
    % print -dpng -r300 -painters ../../Figs_GSPT/paper/MM1_SIM_l.png

%% Visualization of training/testing data
elseif fPlot == 2
    inSz = 2;               % number of variables
    % load training/testing data (on SIM) 
    ft = 1;                 % first index is time
    if fMMcase == 1
        load MM1_TestData allData;
        load MM1_TrainData dataTrain;
        fidx = 1+ft;                      % fast variable index
        sidx = 2+ft;                      % slow variable index
    elseif fMMcase == 2
        load MM2_TestData allData;
        load MM2_TrainData dataTrain;
        fidx = 2+ft;                      % fast variable index
        sidx = 1+ft;                      % slow variable index
    elseif fMMcase == 3
        load MM3_TestData allData;
        load MM3_TrainData dataTrain;
        fidx = 1+ft;                      % fast variable index
        sidx = 2+ft;                      % slow variable index
    end
    XTestData = allData(fidx,:);
    YTestData = allData(sidx,:);
    XTrainData = dataTrain(fidx,:);
    YTrainData = dataTrain(sidx,:);

    % domains Omega for each case
    if fMMcase == 1
        DomC = [1e-5 1; 1e-3 1e+3];              
    elseif fMMcase == 2
        DomC = [1e-3 50; 1e-5 1];                
    elseif fMMcase == 3
        DomC = [1e-5 1; 1e-3 1e+2];               
    end

    % plotting
    figure(fMMcase);
    set(gcf,'renderer','Painters');
    ax = axes('OuterPosition',[0 0 1 1],'PositionConstraint','outerposition');
    scatter(ax,YTestData,XTestData,'r.','DisplayName','Test set'); hold on;
    scatter(ax,YTrainData,XTrainData,24,'bo','LineWidth',0.6,'DisplayName','Train set'); hold off;
    ax.XScale = 'log';
    ax.YScale = 'log';
    if fMMcase == 1      
        offsetx = 0;
        offsety = 1;
        ax.XTick = logspace(-3,3,4);
        ax.YTick = logspace(-4,0,3);
    elseif fMMcase == 2
        offsetx = 0;
        offsety = 1;
        ax.XTick = logspace(-3,1,3);
        ax.YTick = logspace(-4,0,3);
    elseif fMMcase == 3
        offsetx = 0;
        offsety = 1;
        ax.XTick = logspace(-3,1,3);
        ax.YTick = logspace(-4,0,3);
    end
    ax.XLim = [DomC(sidx-ft,1) DomC(sidx-ft,2)+offsetx];
    ax.YLim = [DomC(fidx-ft,1) DomC(fidx-ft,2)+offsety];
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';
    ax.Box = 'on';

    ax.FontName = 'times';
    ax.FontSize = 20;
    ax.LabelFontSizeMultiplier = 24/20;
    ax.TickLabelInterpreter = 'latex';
    if (fMMcase == 1) || (fMMcase == 3)
        ax.XLabel.String = '$s$';
        ax.YLabel.String = '$c$';
    elseif fMMcase == 2
        ax.XLabel.String = '$c$';
        ax.YLabel.String = '$s$';
    end
    ax.XLabel.Interpreter = 'latex';
    ax.XLabel.FontSize = 24;
    ax.YLabel.Interpreter = 'latex';
    ax.YLabel.FontSize = 24;
    ax.YLabel.Rotation = 0;

    lgd = legend;
    lgd.FontName = 'times';
    lgd.FontSize = 20;
    lgd.Interpreter = 'latex';
    lgd.Location = 'southeast';

    prePosition = ax.Position;
    if fMMcase == 1
        ax.XLabel.Position(1) = 1.0;
        ax.XLabel.Position(2) = 3e-6;  
        ax.YLabel.Position(1) = 1.2e-4;
        ax.YLabel.Position(2) = 6e-4;
    elseif fMMcase == 2
        ax.XLabel.Position(1) = 1.0;
        ax.XLabel.Position(2) = 3e-6;  
        ax.YLabel.Position(1) = 1.9e-4;
        ax.YLabel.Position(2) = 6e-4;
    elseif fMMcase == 3
        ax.XLabel.Position(1) = 1.0;
        ax.XLabel.Position(2) = 3e-6;  
        ax.YLabel.Position(1) = 1.8e-4;
        ax.YLabel.Position(2) = 6e-4;
    end
    ax.Position  = prePosition;
    set(gcf,'PaperPositionMode',"auto");

    % inset figure to show Zoom
    ax1 = axes('OuterPosition',[0.15 0.57 0.33 0.33],'PositionConstraint','outerposition');
    scatter(ax1,YTestData,XTestData,'r.','DisplayName','Test set'); hold on;
    scatter(ax1,YTrainData,XTrainData,24,'bo','LineWidth',0.6,'DisplayName','Train set'); hold off;
    ax1.XScale = 'linear';
    ax1.YScale = 'linear';
    if fMMcase == 1      
        ax1.XLim = [5 10];
        ax1.YLim = [4e-2 9e-2];
        ax1.XTick = 6:2:10;
        ax1.YTick = 4e-2:2e-2:8e-2;
    elseif fMMcase == 2
        ax1.XLim = [0.5 1];
        ax1.YLim = [5e-3 1e-2];
        ax1.XTick = 0.6:0.2:1;
    elseif fMMcase == 3
        ax1.XLim = [5 10];
        ax1.YLim = [0.05 0.1];
        ax1.YTick = 0.06:0.02:0.1;
    end
%     
    ax1.Box = 'on';
    ax1.XMinorTick = 'off';
    ax1.YMinorTick = 'off';
    ax1.FontName = 'times';
    ax1.FontSize = 15;
    ax1.LabelFontSizeMultiplier = 24/20;
    ax1.TickLabelInterpreter = 'latex';

    return
    % print -depsc -tiff -r300 -painters ../../Figs_GSPT/MM1_SIM.eps
    % print -dpng -r300 -painters ../../Figs_GSPT/paper/MM1_DataSets_l.png
elseif fPlot == 3
    inSz = 2;               % number of variables
    % load testing data (on SIM) 
    ft = 1;                 % first index is time
    if fMMcase == 1
        load MM1_TestData allData;
        fidx = 1+ft;                      % fast variable index
        sidx = 2+ft;                      % slow variable index
    elseif fMMcase == 2
        load MM2_TestData allData;
        fidx = 2+ft;                      % fast variable index
        sidx = 1+ft;                      % slow variable index 
    elseif fMMcase == 3
        load MM3_TestData allData;
        fidx = 1+ft;                      % fast variable index
        sidx = 2+ft;                      % slow variable index
    end
    Xdata = allData(fidx,:);
    Ydata = allData(sidx,:);

    % parameter sets
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
        DomC = [1e-5 1; 1e-3 1e+2];                % domain Omega to collect collocation points
    end

    % load trained parameters
    netDim = [2 20 1 2 20 1 1 20 1];
    if fMMcase == 1
        load learned_SLFNN_MM1 bestLearned;
    elseif fMMcase == 2
        load learned_SLFNN_MM2 bestLearned;
    elseif fMMcase == 3
        load learned_SLFNN_MM3 bestLearned;
    end

    % find errors of explicit SIM expressions
    % PIML
    Xz = Xtrans(allData(2:3,:),bestLearned,netDim);
    Yz = Ytrans(allData(2:3,:),bestLearned,netDim);
    PIML_SIM = forwardNN3(Yz,bestLearned,netDim);
    AErr_PIML = abs(PIML_SIM-Xz);
    % GSPT expressions
    pars = [k1f k1b k2 e0];
    fImpNR = true;  
    [sQSSA_SIM, rQSSA_SIM, PEA_SIM, CSPe_SIM, CSPc11_SIM, CSPc21_SIM, ...
        CSPs11_SIM, CSPs21_SIM] = MM_knownSIMs(allData(2:3,:),pars,fidx-1,fImpNR);
    AErr_sQSSA = abs(sQSSA_SIM-allData(fidx,:));
    AErr_rQSSA = abs(rQSSA_SIM-allData(fidx,:));
    AErr_PEA = abs(PEA_SIM-allData(fidx,:));
    AErr_CSPe = abs(CSPe_SIM-allData(fidx,:));
    AErr_CSPc11 = abs(CSPc11_SIM-allData(fidx,:));
    AErr_CSPc21 = abs(CSPc21_SIM-allData(fidx,:));
    AErr_CSPs11 = abs(CSPs11_SIM-allData(fidx,:));
    AErr_CSPs21 = abs(CSPs21_SIM-allData(fidx,:));
    % make exact zeros the next higher value 
    AErr_PIML = findZTR(AErr_PIML);
    AErr_sQSSA = findZTR(AErr_sQSSA);
    AErr_rQSSA = findZTR(AErr_rQSSA);
    AErr_PEA = findZTR(AErr_PEA);
    AErr_CSPe = findZTR(AErr_CSPe);
    AErr_CSPc11 = findZTR(AErr_CSPc11);
    AErr_CSPc21 = findZTR(AErr_CSPc21);
    AErr_CSPs11 = findZTR(AErr_CSPs11);
    AErr_CSPs21 = findZTR(AErr_CSPs21);

    % max and min for colorbar
    minAE = min([AErr_PIML AErr_sQSSA AErr_rQSSA AErr_PEA AErr_CSPe AErr_CSPc11 AErr_CSPc21 AErr_CSPs11 AErr_CSPs21]);
    maxAE = max([AErr_PIML AErr_sQSSA AErr_rQSSA AErr_PEA AErr_CSPe AErr_CSPc11 AErr_CSPc21 AErr_CSPs11 AErr_CSPs21]);
    minAE = ceil(log10(minAE));
    maxAE = floor(log10(maxAE));

    % ploting
    figure(fSIMapp);
    set(gcf,'renderer','Painters');
    ax = axes('OuterPosition',[0 0 1 1],'PositionConstraint','outerposition');
    if fSIMapp == 1
        scatter(ax,Ydata(1,:),Xdata(1,:),50,log10(AErr_PIML),'.');
    elseif fSIMapp == 2
        scatter(ax,Ydata(1,:),Xdata(1,:),50,log10(AErr_sQSSA),'.');
    elseif fSIMapp == 3
        scatter(ax,Ydata(1,:),Xdata(1,:),50,log10(AErr_rQSSA),'.');
    elseif fSIMapp == 4
        scatter(ax,Ydata(1,:),Xdata(1,:),50,log10(AErr_PEA),'.');
    elseif fSIMapp == 5
        scatter(ax,Ydata(1,:),Xdata(1,:),50,log10(AErr_CSPe),'.');
    elseif fSIMapp == 6
        scatter(ax,Ydata(1,:),Xdata(1,:),50,log10(AErr_CSPc11),'.');
    elseif fSIMapp == 7
        scatter(ax,Ydata(1,:),Xdata(1,:),50,log10(AErr_CSPc21),'.');
    elseif fSIMapp == 8
        scatter(ax,Ydata(1,:),Xdata(1,:),50,log10(AErr_CSPs11),'.');
    elseif fSIMapp == 9
        scatter(ax,Ydata(1,:),Xdata(1,:),50,log10(AErr_CSPs21),'.');
    end
    ax.XScale = 'log';
    ax.YScale = 'log';
    if fMMcase == 1      
        offsetx = 0;
        offsety = 1;
        ax.XTick = logspace(-3,3,4);
        ax.YTick = logspace(-4,0,3);
    elseif fMMcase == 2
        offsetx = 0;
        offsety = 1;
        ax.XTick = logspace(-3,1,3);
        ax.YTick = logspace(-4,0,3);
    elseif fMMcase == 3
        offsetx = 0;
        offsety = 1;
        ax.XTick = logspace(-3,1,3);
        ax.YTick = logspace(-4,0,3);
    end
    ax.XLim = [DomC(sidx-ft,1) DomC(sidx-ft,2)+offsetx];
    ax.YLim = [DomC(fidx-ft,1) DomC(fidx-ft,2)+offsety];
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';
    ax.Box = 'on';

    % colorbar
    c = colorbar;
    colormap(jet)
    caxis(ax,[minAE maxAE]);
    if (fMMcase == 1) || (fMMcase == 2)
        c.Ticks = -16:3:-4;
    elseif fMMcase ==3
        caxis(ax,[minAE maxAE+1]);
        c.Ticks = -15:4:1;
    end
    c.Label.Interpreter = 'latex';
    c.Label.String = 'log(ae$^{(i)}$)';
    c.Label.FontSize = 20;

    ax.FontName = 'times';
    ax.FontSize = 20;
    ax.LabelFontSizeMultiplier = 24/20;
    ax.TickLabelInterpreter = 'latex';
    ax.XLabel.Interpreter = 'latex';
    ax.XLabel.FontSize = 24;
    ax.YLabel.Interpreter = 'latex';
    ax.YLabel.FontSize = 24;
    if (fMMcase == 1) || (fMMcase == 3)
        ax.XLabel.String = '$s$';
        ax.YLabel.String = '$c$';
    elseif fMMcase == 2
        ax.XLabel.String = '$c$';
        ax.YLabel.String = '$s$';
    end
    ax.YLabel.Rotation = 0;
    
    prePosition = ax.Position;
    if fMMcase == 1
        ax.XLabel.Position(1) = 1.0;
        ax.XLabel.Position(2) = 3e-6;  
        ax.YLabel.Position(1) = 0.9e-4;
        ax.YLabel.Position(2) = 6e-4;
    elseif fMMcase == 2
        ax.XLabel.Position(1) = 1.0;
        ax.XLabel.Position(2) = 3e-6;  
        ax.YLabel.Position(1) = 1.6e-4;
        ax.YLabel.Position(2) = 6e-4;
    elseif fMMcase == 3
        ax.XLabel.Position(1) = 1.0;
        ax.XLabel.Position(2) = 3e-6;  
        ax.YLabel.Position(1) = 1.4e-4;
        ax.YLabel.Position(2) = 6e-4;
    end
    ax.Position  = prePosition;

    return
    % print -depsc -tiff -r300 -painters ../../Figs_paper2/MM1_AE_PIML.eps

end


%%%%%%%% FUNCTIONS
%% event function to terminate integration when solution goes too low
function [value, isterminal, direction] = defineZero(t,y,lowBound)
    value = [y(1)-lowBound(1); y(2)-lowBound(2)];            % when going below the lower bound         
    isterminal = [1; 1];                             % stop integration in ALL events
    direction = [-1; -1];                            % meeting event 1&2: when decreasing values of z  
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

%% function to unravel learnable parameters per NN requested
% NN1 and NN2 are linear and NN3 is actually a network
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

% function that find zeros in array and replaces the value with the next higher value in array
function arrayX = findZTR(arrayX)
    zerIdx = find(arrayX==0);
    minVal = mink(arrayX,numel(zerIdx)+1);
    arrayX(zerIdx) = minVal(end);
end