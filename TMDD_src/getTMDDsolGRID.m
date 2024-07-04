function outData = getTMDDsolGRID(parVec,nSpecs,nSamples,fPeriod)
% getTMDDsolGRID function to generate TMDD requested data points for given initial conditions and parameters. 
%
%  Inputs:   - parVec: [kon koff kel ksyn kdeg kint L0 R0 RL0] 
%            - nSteps: number of steps per orbit at SIM
%            - nSpecs: number of variables
%            - fPeriod: for P2 or P4
%
%  Outputs:  - outData: vector of (N) x # of samples  ON SIM

if (fPeriod~=2) && (fPeriod~=4); error('I only collect data in P2 and P4 of the TMDD!'); end
%
kon = parVec(1);
koff = parVec(2);
kel = parVec(3);
ksyn = parVec(4);
kdeg = parVec(5);
kint = parVec(6);
L0 = parVec(7);
R0 = parVec(8);
RL0 = parVec(9);
%

%% integrate with events
tend = 5000;
tspan = [0 tend];
opts = odeset('RelTol',1e-10,'AbsTol',1e-10);

y_init = [L0 ; R0; RL0];   
sol = ode15s(@(t,y) TMDDode(t,y,kon,koff,kel,ksyn,kdeg,kint),tspan,y_init,opts);

%% incorporate criterion for fast modes
Rtol = 5e-2;
Atol = 1e-10;
Mf = fastModeCrit(sol.x',sol.y',Rtol,Atol,parVec);  % Mf(:,1) number of fast modes, Mf(:,2) biggest amplitude

%% cut before M = 1 and after M = 1 
Mfchang = diff(Mf(:,1));
idxM_inc = find(Mfchang==1,3,'first')+1;
idxM_dec = find(Mfchang==-1,2,'first');
if fPeriod == 2
    t_start = sol.x(idxM_inc(1,1));  % first point where M = 1
    t_end = sol.x(idxM_dec(1,1));    % one point before going to M = 0
    t_onSIM = linspace(t_start,t_end,nSamples);
    y1 = deval(sol,t_onSIM)';
elseif fPeriod == 4
    t_start = sol.x(idxM_inc(2,1));  % second point where M = 1
    t_end = sol.x(idxM_inc(3,1)-1);  % one point before going to M = 2
    t_onSIM = linspace(t_start,t_end,nSamples);
    y1 = deval(sol,t_onSIM)';
end



%% keep data
outData(1,:) = t_onSIM; 
outData(2:1+nSpecs,:) = y1';

end


function Mf = fastModeCrit(t,y,Rtol,Atol,parVec)
    nSpecs = size(y,2);
    nSamples = size(y,1);
    if (numel(Rtol) ~= 1) && (numel(Rtol)~=nSpecs); error('Specify Rtol: N vector, or scalar'); end
    if (numel(Atol) ~= 1) && (numel(Atol)~=nSpecs); error('Specify Atol: N vector, or scalar'); end
    y_err = Rtol.*y'+Atol;

    kon = parVec(1);
    koff = parVec(2);
    kel = parVec(3);
    ksyn = parVec(4);
    kdeg = parVec(5);
    kint = parVec(6);

    %% get RHS, Jacobi and eigenvectors
    RHS = TMDDode(t,y',kon,koff,kel,ksyn,kdeg,kint);
    Jac = gradTMDDode(t,y',kon,koff,kel,ksyn,kdeg,kint);  
    tmscls = zeros(size(t,1),nSpecs);
    fampl = zeros(size(t,1),nSpecs);
    Mf = zeros(size(t,1),2);
    for i = 1:size(t,1)
        [AA, Lambda] = eig(squeeze(Jac(:,:,i)));
        evals = diag(Lambda);
        if norm(imag(evals))~=0; error('Criterion of fast modes, not implemebted for complex eigs!'); end
        % sort eigenvalues and respective eigenvectors
        [evalsS, idx1] = sort(evals,'descend','ComparisonMethod','abs');
        AA = AA(:,idx1);
        BB = inv(AA);
        tmscls(i,:) = 1./abs(evalsS');
        fampl(i,:) = (BB*RHS(:,i))';
        [~, Mf(i,2)] = max(abs(fampl(i,:)));    % maximum amplitude
        %% criterion of fast modes now
        tryM = 0;
        while tryM < nSpecs
            m1 = tryM+1;            
            y_crit = zeros(size(y,2),1);
            for j = 1:m1
                y_crit = y_crit + AA(:,j)*fampl(i,j);
            end
            if m1+1 <= nSpecs
                y_crit = y_crit*tmscls(i,m1+1);
                %y_err = (tmscls(i,m1)/tmscls(i,m1+1))*y(i,:)'+Atol;
            else
                y_crit = y_crit*tmscls(i,m1)*1e+3;
                %y_err = 1e-3*y(i,:)'+Atol;
            end
            if sum(abs(y_crit)<=y_err(:,i)) == nSpecs
                tryM = tryM+1;
            else
                break;
            end
        end
        Mf(i,1) = tryM;
    end

end
