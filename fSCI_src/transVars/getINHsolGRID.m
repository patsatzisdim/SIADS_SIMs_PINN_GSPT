function outData = getINHsolGRID(parVec,nSpecs,nSamples,LowBound)
% getINHsolGRID function to generate Inhibitor mechanism requested data points for given initial conditions and parameters. 
% The output points should lie on the SIM within the given SIMbounds.
%
%  Inputs:   - parVec: [k1f, k1b, k2, k3f, k3b, k4, e0, s10, c10, s20, c20] 
%            - nSteps: number of steps per orbit at SIM
%            - nSpecs: number of variables
%            - LowBound: a value that should be considered zero and the integration stops
%
%  Outputs:  - outData: vector of (N) x # of samples  ON SIM

%
k1f = parVec(1);
k1b = parVec(2);
k2 = parVec(3);
k3f = parVec(4);
k3b = parVec(5);
k4 = parVec(6);
e0 = parVec(7);
% ICs
s10Bar = parVec(8);
c10 = parVec(9);
s20Bar = parVec(10);
c20 = parVec(11);
%

%% integrate with events
tend = 5000;
tspan = [0 tend];
opts = odeset('RelTol',1e-10,'AbsTol',1e-10,'Events',@(t,y) defineZero(t,y,LowBound));

y_init = [s10Bar ; c10; s20Bar ; c20];   
sol = ode15s(@(t,y) INHode(t,y,k1f,k1b,k2,k3f,k3b,k4,e0),tspan,y_init,opts);

%% incorporate criterion for fast modes
Rtol = 5e-2;
Atol = 1e-10;
Mf = fastModeCrit(sol.x',sol.y',Rtol,Atol,parVec);  % Mf(:,1) number of fast modes, Mf(:,2) biggest amplitude

%% cut before M = 2 
idxM2 = find(Mf(:,1)==2,1,'first');
t_start = sol.x(idxM2);  % first point where M = 2
t_end = sol.x(end);    
t_onSIM = linspace(t_start,t_end,nSamples);
y1 = deval(sol,t_onSIM)';

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

    k1f = parVec(1);
    k1b = parVec(2);
    k2 = parVec(3);
    k3f = parVec(4);
    k3b = parVec(5);
    k4 = parVec(6);
    e0 = parVec(7);

    %% get RHS, Jacobi and eigenvectors
    RHS = INHode(t,y',k1f,k1b,k2,k3f,k3b,k4,e0);
    Jac = gradINHode(t,y',k1f,k1b,k2,k3f,k3b,k4,e0);  
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

%% event function to terminate integration when solution goes too low
function [value, isterminal, direction] = defineZero(t,y,lowBound)
    value = [y(1)-lowBound; y(2)-lowBound; y(3)-lowBound; y(4)-lowBound];  % when going below the lower bound         
    isterminal = [1; 1; 1; 1];                                             % stop integration in ALL events
    direction = [-1; -1; -1; -1];                                          % meeting events : when decreasing values of z  
end

