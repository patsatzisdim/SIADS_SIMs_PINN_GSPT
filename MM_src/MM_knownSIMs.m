% function including the GSPT SIM approximations of the MM mechanism
% 
%  Inputs:  - zData: the data points 
%           - pars: model parameters (k1f, k1b, k2, e0)
%           - z2sol: according to which components of z to solve the SIM expressions (applies to explicit 
%             expressions, and to the implicit ones, when the fImpNR is activated)
%           - fImpNR: true (applies NR for solving numerically the implicit expressions), false (gives implicit Error)   
%
%  Outputs: - SIM approximations of the sQSSA, rQSSA, PEA, CSPe, CSPc11, CSPc21, CSPs11, CSPs21 expressions
%
function [sQSSA_SIM, rQSSA_SIM, PEA_SIM, CSPe_SIM, CSPc11_SIM, CSPc21_SIM, ...
                                                CSPs11_SIM, CSPs21_SIM] = MM_knownSIMs(zData,pars,z2sol,fImpNR)
    % pars
    k1f = pars(1);
    k1b = pars(2);
    k2 = pars(3);
    e0 = pars(4);
    %
    c = zData(1,:);
    s = zData(2,:);
    %
    K = k2/k1f;
    KR = k1b/k1f;
    KM = KR+K;
    mu = K./(KR+s);
    nu = (e0-c)./(KR+s);

    %% Solve SIM expressions for the fast variable (if Explicit)
    if z2sol == 1     % here for c
        % QSSAs
        sQSSA_SIM = e0*s./(KM+s); 
        rQSSA_SIM = (e0*k1f*s)./(k1b + k1f*s);
        % PEA
        PEA_SIM = (-k1b^2-k1b*k2-k1b*k1f*e0-2*k1b*k1f*s-k1f*k2*s-2*k1f^2*e0*s-... 
            k1f^2*s.^2+sqrt((k1b^2+k1b*k2+k1b*k1f*e0+2*k1b*k1f*s+k1f*k2*s+... 
            2*k1f^2*e0*s+k1f^2*s.^2).^2-4*(-(k1b*k1f)-k1f^2*s).*(-(k1b*k1f*e0*s)-...
            k1f^2*e0^2*s-k1f^2*e0*s.^2)))./(2*(-(k1b*k1f)-k1f^2*s));     
        % CSP-eigen
        if ~fImpNR
            epsilon = mu.*nu./((1+mu+nu).^2);
            delta = 1-sqrt(1-4*epsilon);
            Ffac = 1-delta.*nu./(2*epsilon.*(1+mu+nu));
            CSPe_SIM = (e0-c).*s-KR*c-K*c.*Ffac;                     % implicit f(c,s)
        else
            CSPe_SIM = zeros(size(c));   % same dimensions
            x_init = e0*0.9; % not e0, cause NaN
            tic
            for i = 1:size(zData,2)
                CSPe_SIM(:,i) = rootNR(@CSPe_imp,s(1,i),pars,x_init);
            end
            CPUend = toc;
            fprintf('Solving CSPe for fast, took NR %e   \n',CPUend);
        end
        % CSPc: (1,1) and (2,1)
        CSPc11_SIM = (k1b^2 + e0*k1b*k1f + 2*k1b*k2 + k2^2 + 2*k1b*k1f*s + 2*e0*k1f^2*s + 2*k1f*k2*s + ...
            k1f^2*s.^2 - sqrt(-4*e0*k1f^2*s.*(k1b + k1f*s).*(k1b + k2 + k1f*(e0 + s)) + (k1b^2 + e0*k1b*k1f + ...
            2*e0*k1f^2*s + 2*k1b*(k2 + k1f*s) + (k2 + k1f*s).^2).^2))./(2*k1f*(k1b + k1f*s)); 
        if ~fImpNR
            CSPc21_SIM = ((k1b + k2 + k1f*s - ((c - e0)*k1f.*(k1b + k1f*s))./(k1b + k2 + k1f*s)).*(-((-k1b - ...
                k2 - k1f*s + ((c - e0)*k1f.*(k1b + k1f*s))./(k1b + k2 + k1f*s)).*(e0*k1f*s - c.*(k1b + k2 + k1f*s))) + ...
                (k1f*(e0*k1f*s - c.*(k1b + k1f*s)).*(c.^2*k1f*k2 + e0*((k1b + k2)*(k1b + e0*k1f + k2) + 3*k1f*(k1b + ...
                k2)*s + 2*k1f^2*s.^2) - c.*(2*k1b^2 + e0*k1b*k1f + 2*e0*k1f*k2 + 4*k1b*(k2 + k1f*s) + 2*(k2 + ...
                k1f*s).^2)))./(k1b + k2 + k1f*s).^2))./(k1b^2 + 2*k1b*k2 + (-c + e0)*k1f^2.*s + k1b*k1f*(-c + e0 + ...
                2*s) + (k2 + k1f*s).^2 + (k1f*(k1b + k1f*s).*(c.^2*k1f*k2 + e0*((k1b + k2)*(k1b + e0*k1f + k2) + ...
                3*k1f*(k1b + k2)*s + 2*k1f^2*s.^2) - c.*(2*k1b^2 + e0*k1b*k1f + 2*e0*k1f*k2 + 4*k1b*(k2 + k1f*s) + ...
                2*(k2 + k1f*s).^2)))./(k1b + k2 + k1f*s).^2);
        else
            CSPc21_SIM = zeros(size(c));   % same dimensions
            x_init = e0*0.9; % not e0, cause NaN
            tic
            for i = 1:size(zData,2)
                CSPc21_SIM(:,i) = rootNR(@CSPc21_impC,s(1,i),pars,x_init);
            end
            CPUend = toc;
            fprintf('Solving CSPc21 for fast, took NR %e   \n',CPUend);
        end
        % CSPs: (1,1) and (2,1)
        CSPs11_SIM = (k1b^2 + e0*k1b*k1f + k1b*k2 + 2*k1b*k1f*s + 2*e0*k1f^2*s + k1f*k2*s + k1f^2*s.^2 - sqrt((-k1b^2 - ...
            e0*k1b*k1f - k1b*k2 - 2*k1b*k1f*s - 2*e0*k1f^2*s - k1f*k2*s - k1f^2*s.^2).^2 - 4*(k1b*k1f + k1f^2*s).*(e0*k1b*k1f*s + ...
            e0^2*k1f^2*s + e0*k1f^2*s.^2)))./(2*(k1b*k1f + k1f^2*s));
        if ~fImpNR
            CSPs21_SIM = ((k1b + k1f*(-c + e0 + s)).*(-((c - e0).^2.*(k1b + k1f*(-c + e0 + s)).*(-(e0*k1f*s) + c.*(k1b + k1f*s))) + ...
                ((e0*k1f*s - c.*(k1b + k2 + k1f*s)).*(e0*k1b*(k1b - 3*c*k1f + k2) + e0*k1f*(k1b - 4*c*k1f + k2).*s + 2*c.^2*k1f.*(k1b + ...
                k1f*s) + e0^2*k1f*(k1b + 2*k1f*s)))/k1f))./((c - e0).*(e0^3*k1f^2 + c.^2*k1f.*(3*k1b - c*k1f + 3*k1f*s) + e0*(k1b^2 - ...
                5*c*k1b*k1f + 3*c.^2*k1f^2 + k1b*k2 + k1f*(k1b - 6*c*k1f + k2).*s) + e0^2*k1f*(2*k1b + 3*k1f*(-c + s)))); 
        else
            CSPs21_SIM = zeros(size(c));   % same dimensions
            x_init = e0*0.9; % not e0, cause NaN
            tic
            for i = 1:size(zData,2)
                CSPs21_SIM(:,i) = rootNR(@CSPs21_imp,s(1,i),pars,x_init);
            end
            CPUend = toc;
            fprintf('Solving CSPs21 for fast, took NR %e   \n',CPUend);
        end

    elseif z2sol == 2     % here for s
        sQSSA_SIM = -(c*(k2 + k1b))./(k1f*(c - e0));
        rQSSA_SIM = -(c*k1b)./(k1f*(c - e0));
        PEA_SIM = -(sqrt(c.^4*k1f^2 - 4*c.^3*e0*k1f^2 - 2*c.^3*k2*k1f + 6*c.^2*e0^2*k1f^2 + 4*c.^2*e0*k2*k1f + ...
            2*c.^2*e0*k1b*k1f + c.^2*k2^2 - 4*c*e0^3*k1f^2 - 2*c*e0^2*k2*k1f - 4*c*e0^2*k1b*k1f + 2*c*e0*k2*k1b + ...
            e0^4*k1f^2 + 2*e0^3*k1b*k1f + e0^2*k1b^2) + c*k2 + 2*c*k1b - e0*k1b - c.^2*k1f - e0^2*k1f + ...
            2*c*e0*k1f)./(2*k1f*(c - e0));
        CSPc11_SIM = (c.^2*k1f - 2*c*(k1b + e0*k1f + k2) + e0*(k1b + e0*k1f + k2) - sqrt((e0*k1b + (c - e0).^2*k1f).^2 + ... 
            2*(e0^2*k1b + (c - e0).^2.*(-2*c + e0)*k1f)*k2 + e0^2*k2^2))./(2.*(c - e0)*k1f);
        if ~fImpNR                                                                                                                                                                                   
            CSPc21_SIM = ((k1b + k2 + k1f*s - ((c - e0)*k1f.*(k1b + k1f*s))./(k1b + k2 + k1f*s)).*(-((-k1b - ...
                k2 - k1f*s + ((c - e0)*k1f.*(k1b + k1f*s))./(k1b + k2 + k1f*s)).*(e0*k1f*s - c.*(k1b + k2 + k1f*s))) + ...
                (k1f*(e0*k1f*s - c.*(k1b + k1f*s)).*(c.^2*k1f*k2 + e0*((k1b + k2)*(k1b + e0*k1f + k2) + 3*k1f*(k1b + ...
                k2)*s + 2*k1f^2*s.^2) - c.*(2*k1b^2 + e0*k1b*k1f + 2*e0*k1f*k2 + 4*k1b*(k2 + k1f*s) + 2*(k2 + ...
                k1f*s).^2)))./(k1b + k2 + k1f*s).^2))./(k1b^2 + 2*k1b*k2 + (-c + e0)*k1f^2.*s + k1b*k1f*(-c + e0 + ...
                2*s) + (k2 + k1f*s).^2 + (k1f*(k1b + k1f*s).*(c.^2*k1f*k2 + e0*((k1b + k2)*(k1b + e0*k1f + k2) + ...
                3*k1f*(k1b + k2)*s + 2*k1f^2*s.^2) - c.*(2*k1b^2 + e0*k1b*k1f + 2*e0*k1f*k2 + 4*k1b*(k2 + k1f*s) + ...
                2*(k2 + k1f*s).^2)))./(k1b + k2 + k1f*s).^2);
        else
            CSPc21_SIM = zeros(size(s));   % same dimensions
            x_init = 0;
            tic
            for i = 1:size(zData,2)
                CSPc21_SIM(:,i) = rootNR(@CSPc21_impS,c(1,i),pars,x_init);
            end
            CPUend = toc;
            fprintf('Solving CSPc21 for fast, took NR %e   \n',CPUend);
        end
        CSPs11_SIM = -(sqrt(c.^4*k1f^2 - 4*c.^3*e0*k1f^2 - 2*c.^3*k2*k1f + 6*c.^2*e0^2*k1f^2 + 4*c.^2*e0*k2*k1f + ...
            2*c.^2*e0*k1b*k1f + c.^2*k2^2 - 4*c*e0^3*k1f^2 - 2*c*e0^2*k2*k1f - 4*c*e0^2*k1b*k1f + 2*c*e0*k2*k1b + ...
            e0^4*k1f^2 + 2*e0^3*k1b*k1f + e0^2*k1b^2) + c*k2 + 2*c*k1b - e0*k1b - c.^2*k1f - e0^2*k1f + ...
            2*c*e0*k1f)./(2*k1f*(c - e0));
        CSPs21_SIM= ((c - e0).^4*k1f^3 + e0*k1f*(k1b + k2)*(e0*k1b - c*(2*k1b + k2)) - 2*(c - e0).^2*k1f^2.*(-(e0*k1b) + ...
            c*(3*k1b + k2)) - sqrt(k1f^2*(4*c.*(c - e0)*k1b.*(3*c.^2*k1f + e0*(k1b - 6*c*k1f + 3*e0*k1f + k2)).*(-(e0^3*k1f^2) + ...
            c.^2*k1f.*(-3*k1b + c*k1f - 2*k2) - e0^2*k1f*(2*k1b - 3*c*k1f + k2) - e0*(k1b^2 - 5*c*k1b*k1f + 3*c.^2*k1f^2 + ...
            2*k1b*k2 - 3*c*k1f*k2 + k2^2)) + (c.^4*k1f^2 - 2*c.^3*k1f*(3*k1b + 2*e0*k1f + k2) + 2*c.^2*e0*k1f*(7*k1b + 3*e0*k1f + ...
            2*k2) + e0^2*((k1b + e0*k1f)^2 + k1b*k2) - c*e0*(2*k1b^2 + 10*e0*k1b*k1f + 4*e0^2*k1f^2 + 3*k1b*k2 + 2*e0*k1f*k2 + ...
            k2^2)).^2)))./(2.*(c - e0)*k1f^2.*(3*c.^2*k1f + e0*(k1b - 6*c*k1f + 3*e0*k1f + k2))); 
        CSPe_SIM = -(c.*(sqrt(c.^4*k1f^2 - 4*c.^3*e0*k1f^2 + 6*c.^2*e0^2*k1f^2 - 2*c.^2*e0*k2*k1f + 2*c.^2*e0*k1b*k1f - ...
            4*c*e0^3*k1f^2 + 4*c*e0^2*k2*k1f - 4*c*e0^2*k1b*k1f + e0^4*k1f^2 - 2*e0^3*k2*k1f + 2*e0^3*k1b*k1f + ...
            e0^2*k2^2 + 2*e0^2*k2*k1b + e0^2*k1b^2) + e0*k2 + e0*k1b - c.^2*k1f - e0^2*k1f + 2*c*e0*k1f))./(2*e0*k1f*(c - e0));
    end



end

%%% Newton-Raphson iterative scheme %%% 
function xNR = rootNR(funHandle,ySlow,pars,x0)
    tol = 1e-6;
    dx = 100;
    iter = 0;
    while (abs(dx)>tol) && (iter<10)
        [fun, dfun_dx] = funHandle(x0,ySlow,pars);
        dx = -fun/dfun_dx;
        % Newton Update
        x0 = x0 + dx;
        iter = iter + 1;
        if iter == 10
            error('I want more iterations, Newton said \n');
        end
    end
    xNR = x0;

end


%%% FUNCTION handlers to use in NR %%%
%
%
%

% CSP with eigenvectors
function [CSPe_SIM, CSPe_SIM_dc] = CSPe_imp(xFast,ySlow,pars)
    k1f = pars(1);
    k1b = pars(2);
    k2 = pars(3);
    e0 = pars(4);
    % in this case 
    c = xFast;
    s = ySlow;
    % 
    K = k2/k1f;
    KR = k1b/k1f;
    mu = K./(KR+s);
    nu = (e0-c)./(KR+s);
    epsilon = mu.*nu./((1+mu+nu).^2);
    delta = 1-sqrt(1-4*epsilon);
    Ffac = 1-delta.*nu./(2*epsilon.*(1+mu+nu));
    CSPe_SIM = (e0-c).*s-KR*c-K*c.*Ffac;                     % implicit f(c,s)
    %
    CSPe_SIM_dc = (c*k2*((k1f*(((c^2*k1f^2 - 2*c*e0*k1f^2 + 2*c*k2*k1f - 2*c*k1b*k1f - 2*c*k1f^2*s + ...
        e0^2*k1f^2 - 2*e0*k2*k1f + 2*e0*k1b*k1f + 2*e0*k1f^2*s + k2^2 + 2*k2*k1b + 2*k2*k1f*s + k1b^2 + ...
        2*k1b*k1f*s + k1f^2*s^2)/(k2 + k1b - c*k1f + e0*k1f + k1f*s)^2)^(1/2) - 1))/(2*k2) - (k1f*(k2 + k1b + c*k1f - ...
        e0*k1f + k1f*s))/(((c^2*k1f^2 - 2*c*e0*k1f^2 + 2*c*k2*k1f - 2*c*k1b*k1f - 2*c*k1f^2*s + e0^2*k1f^2 - ...
        2*e0*k2*k1f + 2*e0*k1b*k1f + 2*e0*k1f^2*s + k2^2 + 2*k2*k1b + 2*k2*k1f*s + k1b^2 + 2*k1b*k1f*s + k1f^2*s^2)/(k2 + ...
        k1b - c*k1f + e0*k1f + k1f*s)^2)^(1/2)*(k2 + k1b - c*k1f + e0*k1f + k1f*s)^2)))/k1f - k1b/k1f - ...
        (k2*(((((c^2*k1f^2 - 2*c*e0*k1f^2 + 2*c*k2*k1f - 2*c*k1b*k1f - 2*c*k1f^2*s + e0^2*k1f^2 - 2*e0*k2*k1f + ...
        2*e0*k1b*k1f + 2*e0*k1f^2*s + k2^2 + 2*k2*k1b + 2*k2*k1f*s + k1b^2 + 2*k1b*k1f*s + k1f^2*s^2)/(k2 + k1b - c*k1f + ...
        e0*k1f + k1f*s)^2)^(1/2) - 1)*(k2 + k1b - c*k1f + e0*k1f + k1f*s))/(2*k2) + 1))/k1f - s;
end

% CSPc21 solved for c
function [CSPc21_SIM, CSPc21_SIM_dc] = CSPc21_impC(xFast,ySlow,pars)
    k1f = pars(1);
    k1b = pars(2);
    k2 = pars(3);
    e0 = pars(4);
    % in this case 
    c = xFast;
    s = ySlow;
    % 
    CSPc21_SIM = ((k1b + k2 + k1f*s - ((c - e0)*k1f.*(k1b + k1f*s))./(k1b + k2 + k1f*s)).*(-((-k1b - ...
        k2 - k1f*s + ((c - e0)*k1f.*(k1b + k1f*s))./(k1b + k2 + k1f*s)).*(e0*k1f*s - c.*(k1b + k2 + k1f*s))) + ...
        (k1f*(e0*k1f*s - c.*(k1b + k1f*s)).*(c.^2*k1f*k2 + e0*((k1b + k2)*(k1b + e0*k1f + k2) + 3*k1f*(k1b + ...
        k2)*s + 2*k1f^2*s.^2) - c.*(2*k1b^2 + e0*k1b*k1f + 2*e0*k1f*k2 + 4*k1b*(k2 + k1f*s) + 2*(k2 + ...
        k1f*s).^2)))./(k1b + k2 + k1f*s).^2))./(k1b^2 + 2*k1b*k2 + (-c + e0)*k1f^2.*s + k1b*k1f*(-c + e0 + ...
        2*s) + (k2 + k1f*s).^2 + (k1f*(k1b + k1f*s).*(c.^2*k1f*k2 + e0*((k1b + k2)*(k1b + e0*k1f + k2) + ...
        3*k1f*(k1b + k2)*s + 2*k1f^2*s.^2) - c.*(2*k1b^2 + e0*k1b*k1f + 2*e0*k1f*k2 + 4*k1b*(k2 + k1f*s) + ...
        2*(k2 + k1f*s).^2)))./(k1b + k2 + k1f*s).^2);
    %
    CSPc21_SIM_dc = (-((k1b + k2 + k1f*s - ((c - e0)*k1f*(k1b + k1f*s))/(k1b + k2 + k1f*s))*(k1f*(e0*k1f*s - ...
        c*(k1b + k1f*s))*(2*k1b^2 + e0*k1b*k1f - 2*c*k1f*k2 + 2*e0*k1f*k2 + 4*k1b*(k2 + k1f*s) + 2*(k2 + k1f*s)^2) + ...
        k1f*(k1b + k1f*s)*(k1b + k2 + k1f*s)*(e0*k1f*s - c*(k1b + k2 + k1f*s)) + (k1b + k2 + k1f*s)^2*(-((c - e0)*k1f*(k1b + ...
        k1f*s)) + k1b*(k1b + k2 + k1f*s) + k2*(k1b + k2 + k1f*s) + k1f*s*(k1b + k2 + k1f*s)) + k1f*(k1b + k1f*s)*(c^2*k1f*k2 + ...
        e0*((k1b + k2)*(k1b + e0*k1f + k2) + 3*k1f*(k1b + k2)*s + 2*k1f^2*s^2) - c*(2*k1b^2 + e0*k1b*k1f + 2*e0*k1f*k2 + ...
        4*k1b*(k2 + k1f*s) + 2*(k2 + k1f*s)^2)))*(k1b^2 + 2*k1b*k2 + (-c + e0)*k1f^2*s + k1b*k1f*(-c + e0 + 2*s) + (k2 + ...
        k1f*s)^2 + (k1f*(k1b + k1f*s)*(c^2*k1f*k2 + e0*((k1b + k2)*(k1b + e0*k1f + k2) + 3*k1f*(k1b + k2)*s + 2*k1f^2*s^2) - ...
        c*(2*k1b^2 + e0*k1b*k1f + 2*e0*k1f*k2 + 4*k1b*(k2 + k1f*s) + 2*(k2 + k1f*s)^2)))/(k1b + k2 + k1f*s)^2)) + k1f*(k1b + ...
        k1f*s)*(3*k1b^2 + e0*k1b*k1f - 2*c*k1f*k2 + 2*e0*k1f*k2 + 3*k2^2 + 6*k1f*k2*s + 3*k1f^2*s^2 + 6*k1b*(k2 + k1f*s))*(k1b + ...
        k2 + k1f*s - ((c - e0)*k1f*(k1b + k1f*s))/(k1b + k2 + k1f*s))*(-((-k1b - k2 - k1f*s + ((c - e0)*k1f*(k1b + k1f*s))/(k1b + ...
        k2 + k1f*s))*(e0*k1f*s - c*(k1b + k2 + k1f*s))) + (k1f*(e0*k1f*s - c*(k1b + k1f*s))*(c^2*k1f*k2 + e0*((k1b + k2)*(k1b + ...
        e0*k1f + k2) + 3*k1f*(k1b + k2)*s + 2*k1f^2*s^2) - c*(2*k1b^2 + e0*k1b*k1f + 2*e0*k1f*k2 + 4*k1b*(k2 + k1f*s) + 2*(k2 + ...
        k1f*s)^2)))/(k1b + k2 + k1f*s)^2) - k1f*(k1b + k1f*s)*(k1b + k2 + k1f*s)*(k1b^2 + 2*k1b*k2 + (-c + e0)*k1f^2*s + ...
        k1b*k1f*(-c + e0 + 2*s) + (k2 + k1f*s)^2 + (k1f*(k1b + k1f*s)*(c^2*k1f*k2 + e0*((k1b + k2)*(k1b + e0*k1f + k2) + ...
        3*k1f*(k1b + k2)*s + 2*k1f^2*s^2) - c*(2*k1b^2 + e0*k1b*k1f + 2*e0*k1f*k2 + 4*k1b*(k2 + k1f*s) + 2*(k2 + k1f*s)^2)))/(k1b + ...
        k2 + k1f*s)^2)*(-((-k1b - k2 - k1f*s + ((c - e0)*k1f*(k1b + k1f*s))/(k1b + k2 + k1f*s))*(e0*k1f*s - c*(k1b + k2 + ...
        k1f*s))) + (k1f*(e0*k1f*s - c*(k1b + k1f*s))*(c^2*k1f*k2 + e0*((k1b + k2)*(k1b + e0*k1f + k2) + 3*k1f*(k1b + k2)*s + ...
        2*k1f^2*s^2) - c*(2*k1b^2 + e0*k1b*k1f + 2*e0*k1f*k2 + 4*k1b*(k2 + k1f*s) + 2*(k2 + k1f*s)^2)))/(k1b + k2 + ...
        k1f*s)^2))/((k1b + k2 + k1f*s)^2*(k1b^2 + 2*k1b*k2 + (-c + e0)*k1f^2*s + k1b*k1f*(-c + e0 + 2*s) + (k2 + k1f*s)^2 + ...
        (k1f*(k1b + k1f*s)*(c^2*k1f*k2 + e0*((k1b + k2)*(k1b + e0*k1f + k2) + 3*k1f*(k1b + k2)*s + 2*k1f^2*s^2) - c*(2*k1b^2 + ...
        e0*k1b*k1f + 2*e0*k1f*k2 + 4*k1b*(k2 + k1f*s) + 2*(k2 + k1f*s)^2)))/(k1b + k2 + k1f*s)^2)^2); 
end

% CSPs21 solved for c
function [CSPs21_SIM, CSPs21_SIM_dc] = CSPs21_imp(xFast,ySlow,pars)
    k1f = pars(1);
    k1b = pars(2);
    k2 = pars(3);
    e0 = pars(4);
    % in this case 
    c = xFast;
    s = ySlow;
    % 
    CSPs21_SIM = ((k1b + k1f*(-c + e0 + s)).*(-((c - e0).^2.*(k1b + k1f*(-c + e0 + s)).*(-(e0*k1f*s) + c.*(k1b + k1f*s))) + ...
        ((e0*k1f*s - c.*(k1b + k2 + k1f*s)).*(e0*k1b*(k1b - 3*c*k1f + k2) + e0*k1f*(k1b - 4*c*k1f + k2).*s + 2*c.^2*k1f.*(k1b + ...
        k1f*s) + e0^2*k1f*(k1b + 2*k1f*s)))/k1f))./((c - e0).*(e0^3*k1f^2 + c.^2*k1f.*(3*k1b - c*k1f + 3*k1f*s) + e0*(k1b^2 - ...
        5*c*k1b*k1f + 3*c.^2*k1f^2 + k1b*k2 + k1f*(k1b - 6*c*k1f + k2).*s) + e0^2*k1f*(2*k1b + 3*k1f*(-c + s)))); 
    %
    CSPs21_SIM_dc = -(((c - e0)*(k1b + k1f*(-c + e0 + s))*(e0^3*k1f^2*(k1b + 4*k1f*s) + c^2*k1f*(k1b + k1f*s)*(9*k1b - 4*c*k1f + 6*k2 + ...
        9*k1f*s) + e0^2*k1f*(k1b*(2*k1b - 6*c*k1f + k2) + 2*k1f*(5*k1b - 6*c*k1f + k2)*s + 9*k1f^2*s^2) + e0*(k1b*(k1b^2 + 2*k1b*(-5*c*k1f + ...
        k2) + (-3*c*k1f + k2)^2) + k1f*(2*k1b^2 - 28*c*k1b*k1f + 12*c^2*k1f^2 + 3*k1b*k2 - 8*c*k1f*k2 + k2^2)*s + k1f^2*(k1b - 18*c*k1f + ...
        k2)*s^2))*(e0^3*k1f^2 + c^2*k1f*(3*k1b - c*k1f + 3*k1f*s) + e0*(k1b^2 - 5*c*k1b*k1f + 3*c^2*k1f^2 + k1b*k2 + k1f*(k1b - 6*c*k1f + k2)*s) + ...
        e0^2*k1f*(2*k1b + 3*k1f*(-c + s))) + (c - e0)*k1f*(e0^3*k1f^2 + c^2*k1f*(3*k1b - c*k1f + 3*k1f*s) + e0*(k1b^2 - 5*c*k1b*k1f + 3*c^2*k1f^2 + ...
        k1b*k2 + k1f*(k1b - 6*c*k1f + k2)*s) + e0^2*k1f*(2*k1b + 3*k1f*(-c + s)))*((c - e0)^2*k1f*(k1b + k1f*(-c + e0 + s))*(e0*k1f*s - c*(k1b + k1f*s)) + ...
        (e0*k1f*s - c*(k1b + k2 + k1f*s))*(e0*k1b*(k1b - 3*c*k1f + k2) + e0*k1f*(k1b - 4*c*k1f + k2)*s + 2*c^2*k1f*(k1b + k1f*s) + e0^2*k1f*(k1b + 2*k1f*s))) + ...
        (k1b + k1f*(-c + e0 + s))*(e0^3*k1f^2 + c^2*k1f*(3*k1b - c*k1f + 3*k1f*s) + e0*(k1b^2 - 5*c*k1b*k1f + 3*c^2*k1f^2 + k1b*k2 + k1f*(k1b - 6*c*k1f + k2)*s) + ...
        e0^2*k1f*(2*k1b + 3*k1f*(-c + s)))*((c - e0)^2*k1f*(k1b + k1f*(-c + e0 + s))*(e0*k1f*s - c*(k1b + k1f*s)) + (e0*k1f*s - c*(k1b + k2 + k1f*s))*(e0*k1b*(k1b - 3*c*k1f + k2) + ...
        e0*k1f*(k1b - 4*c*k1f + k2)*s + 2*c^2*k1f*(k1b + k1f*s) + e0^2*k1f*(k1b + 2*k1f*s))) - (c - e0)*k1f*(k1b + k1f*(-c + e0 + s))*(3*c^2*k1f - 6*c*(k1b + k1f*(e0 + s)) + ...
        e0*(5*k1b + 3*k1f*(e0 + 2*s)))*((c - e0)^2*k1f*(k1b + k1f*(-c + e0 + s))*(e0*k1f*s - c*(k1b + k1f*s)) + (e0*k1f*s - c*(k1b + k2 + k1f*s))*(e0*k1b*(k1b - 3*c*k1f + k2) + e0*k1f*(k1b - ...
        4*c*k1f + k2)*s + 2*c^2*k1f*(k1b + k1f*s) + e0^2*k1f*(k1b + 2*k1f*s))))/((c - e0)^2*k1f*(e0^3*k1f^2 + c^2*k1f*(3*k1b - c*k1f + 3*k1f*s) + e0*(k1b^2 - 5*c*k1b*k1f + 3*c^2*k1f^2 + ...
        k1b*k2 + k1f*(k1b - 6*c*k1f + k2)*s) + e0^2*k1f*(2*k1b + 3*k1f*(-c + s)))^2)); 

end

% CSPc21 solved for c
function [CSPc21_SIM, CSPc21_SIM_ds] = CSPc21_impS(xFast,ySlow,pars)
    k1f = pars(1);
    k1b = pars(2);
    k2 = pars(3);
    e0 = pars(4);
    % in this case 
    c = ySlow;
    s = xFast;
    % 
    CSPc21_SIM = ((k1b + k2 + k1f*s - ((c - e0)*k1f.*(k1b + k1f*s))./(k1b + k2 + k1f*s)).*(-((-k1b - ...
        k2 - k1f*s + ((c - e0)*k1f.*(k1b + k1f*s))./(k1b + k2 + k1f*s)).*(e0*k1f*s - c.*(k1b + k2 + k1f*s))) + ...
        (k1f*(e0*k1f*s - c.*(k1b + k1f*s)).*(c.^2*k1f*k2 + e0*((k1b + k2)*(k1b + e0*k1f + k2) + 3*k1f*(k1b + ...
        k2)*s + 2*k1f^2*s.^2) - c.*(2*k1b^2 + e0*k1b*k1f + 2*e0*k1f*k2 + 4*k1b*(k2 + k1f*s) + 2*(k2 + ...
        k1f*s).^2)))./(k1b + k2 + k1f*s).^2))./(k1b^2 + 2*k1b*k2 + (-c + e0)*k1f^2.*s + k1b*k1f*(-c + e0 + ...
        2*s) + (k2 + k1f*s).^2 + (k1f*(k1b + k1f*s).*(c.^2*k1f*k2 + e0*((k1b + k2)*(k1b + e0*k1f + k2) + ...
        3*k1f*(k1b + k2)*s + 2*k1f^2*s.^2) - c.*(2*k1b^2 + e0*k1b*k1f + 2*e0*k1f*k2 + 4*k1b*(k2 + k1f*s) + ...
        2*(k2 + k1f*s).^2)))./(k1b + k2 + k1f*s).^2);
    %
    CSPc21_SIM_ds = (k1f*((k1b + k2 + k1f*s - ((c - e0)*k1f*(k1b + k1f*s))/(k1b + k2 + k1f*s))*(k1b^2 + 2*k1b*k2 + ...
        (-c + e0)*k1f^2*s + k1b*k1f*(-c + e0 + 2*s) + (k2 + k1f*s)^2 + (k1f*(k1b + k1f*s)*(c^2*k1f*k2 + e0*((k1b + ...
        k2)*(k1b + e0*k1f + k2) + 3*k1f*(k1b + k2)*s + 2*k1f^2*s^2) - c*(2*k1b^2 + e0*k1b*k1f + 2*e0*k1f*k2 + ...
        4*k1b*(k2 + k1f*s) + 2*(k2 + k1f*s)^2)))/(k1b + k2 + k1f*s)^2)*(-((-c + e0)*(-k1b - k2 - k1f*s + ((c - e0)*k1f*(k1b + ...
        k1f*s))/(k1b + k2 + k1f*s))) - ((k1b^2 - c*k1f*k2 + e0*k1f*k2 + k2^2 + 2*k1f*k2*s + k1f^2*s^2 + 2*k1b*(k2 + ...
        k1f*s))*(-(e0*k1f*s) + c*(k1b + k2 + k1f*s)))/(k1b + k2 + k1f*s)^2 + (k1f*(-(e0*k1f*s) + c*(k1b + k1f*s))*(4*c*(k1b + ...
        k2 + k1f*s) - e0*(3*k1b + 3*k2 + 4*k1f*s)))/(k1b + k2 + k1f*s)^2 + ((-c + e0)*k1f*(c^2*k1f*k2 + e0*((k1b + k2)*(k1b + ...
        e0*k1f + k2) + 3*k1f*(k1b + k2)*s + 2*k1f^2*s^2) - c*(2*k1b^2 + e0*k1b*k1f + 2*e0*k1f*k2 + 4*k1b*(k2 + k1f*s) + 2*(k2 + ...
        k1f*s)^2)))/(k1b + k2 + k1f*s)^2 - (2*k1f*(e0*k1f*s - c*(k1b + k1f*s))*(c^2*k1f*k2 + e0*((k1b + k2)*(k1b + e0*k1f + k2) + ...
        3*k1f*(k1b + k2)*s + 2*k1f^2*s^2) - c*(2*k1b^2 + e0*k1b*k1f + 2*e0*k1f*k2 + 4*k1b*(k2 + k1f*s) + 2*(k2 + k1f*s)^2)))/(k1b + ...
        k2 + k1f*s)^3) - (k1b + k2 + k1f*s - ((c - e0)*k1f*(k1b + k1f*s))/(k1b + k2 + k1f*s))*(2*k1b + (-c + e0)*k1f + 2*(k2 + ...
        k1f*s) + (k1f*(k1b + k1f*s)*(-4*c*(k1b + k2 + k1f*s) + e0*(3*k1b + 3*k2 + 4*k1f*s)))/(k1b + k2 + k1f*s)^2 - (2*k1f*(k1b + ...
        k1f*s)*(c^2*k1f*k2 + e0*((k1b + k2)*(k1b + e0*k1f + k2) + 3*k1f*(k1b + k2)*s + 2*k1f^2*s^2) - c*(2*k1b^2 + e0*k1b*k1f + ...
        2*e0*k1f*k2 + 4*k1b*(k2 + k1f*s) + 2*(k2 + k1f*s)^2)))/(k1b + k2 + k1f*s)^3 + (k1f*(c^2*k1f*k2 + e0*((k1b + k2)*(k1b + ...
        e0*k1f + k2) + 3*k1f*(k1b + k2)*s + 2*k1f^2*s^2) - c*(2*k1b^2 + e0*k1b*k1f + 2*e0*k1f*k2 + 4*k1b*(k2 + k1f*s) + 2*(k2 + ...
        k1f*s)^2)))/(k1b + k2 + k1f*s)^2)*(-((-k1b - k2 - k1f*s + ((c - e0)*k1f*(k1b + k1f*s))/(k1b + k2 + k1f*s))*(e0*k1f*s - ...
        c*(k1b + k2 + k1f*s))) + (k1f*(e0*k1f*s - c*(k1b + k1f*s))*(c^2*k1f*k2 + e0*((k1b + k2)*(k1b + e0*k1f + k2) + 3*k1f*(k1b + ...
        k2)*s + 2*k1f^2*s^2) - c*(2*k1b^2 + e0*k1b*k1f + 2*e0*k1f*k2 + 4*k1b*(k2 + k1f*s) + 2*(k2 + k1f*s)^2)))/(k1b + k2 + ...
        k1f*s)^2) + (1 + ((c - e0)*k1f*(k1b + k1f*s))/(k1b + k2 + k1f*s)^2 + ((-c + e0)*k1f)/(k1b + k2 + k1f*s))*(k1b^2 + ...
        2*k1b*k2 + (-c + e0)*k1f^2*s + k1b*k1f*(-c + e0 + 2*s) + (k2 + k1f*s)^2 + (k1f*(k1b + k1f*s)*(c^2*k1f*k2 + e0*((k1b + ...
        k2)*(k1b + e0*k1f + k2) + 3*k1f*(k1b + k2)*s + 2*k1f^2*s^2) - c*(2*k1b^2 + e0*k1b*k1f + 2*e0*k1f*k2 + 4*k1b*(k2 + k1f*s) + ...
        2*(k2 + k1f*s)^2)))/(k1b + k2 + k1f*s)^2)*(-((-k1b - k2 - k1f*s + ((c - e0)*k1f*(k1b + k1f*s))/(k1b + k2 + k1f*s))*(e0*k1f*s - ...
        c*(k1b + k2 + k1f*s))) + (k1f*(e0*k1f*s - c*(k1b + k1f*s))*(c^2*k1f*k2 + e0*((k1b + k2)*(k1b + e0*k1f + k2) + 3*k1f*(k1b + ...
        k2)*s + 2*k1f^2*s^2) - c*(2*k1b^2 + e0*k1b*k1f + 2*e0*k1f*k2 + 4*k1b*(k2 + k1f*s) + 2*(k2 + k1f*s)^2)))/(k1b + k2 + ...
        k1f*s)^2)))/(k1b^2 + 2*k1b*k2 + (-c + e0)*k1f^2*s + k1b*k1f*(-c + e0 + 2*s) + (k2 + k1f*s)^2 + (k1f*(k1b + k1f*s)*(c^2*k1f*k2 + ...
        e0*((k1b + k2)*(k1b + e0*k1f + k2) + 3*k1f*(k1b + k2)*s + 2*k1f^2*s^2) - c*(2*k1b^2 + e0*k1b*k1f + 2*e0*k1f*k2 + ...
        4*k1b*(k2 + k1f*s) + 2*(k2 + k1f*s)^2)))/(k1b + k2 + k1f*s)^2)^2;
end