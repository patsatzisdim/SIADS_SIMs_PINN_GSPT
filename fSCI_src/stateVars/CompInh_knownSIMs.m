% function including the known SIMs of the MM mechanism
% 
%  Inputs:  - zData: the data points 
%           - pars: model parameters (k1f, k1b, k2, k3f, k3b, k4, e0)
%           - z2sol: according to which components of z to solve the SIM expressions (applies to explicit 
%             expressions, and to the implicit ones, when the fImpNR is activated)
%           - fImpNR: true (applies NR for solving numerically the implicit expressions), false (gives implicit Error)   
%
%  Outputs: - SIM approximations of the QSSA, PEA12, CSPL11, CSP11, CSP21 expressions for c1,c2 being fast only!!
%
function [QSSAc1c2_SIM, PEA13c1c2_SIM, CSP11c1c2_SIM, CSP21c1c2_SIM] = CompInh_knownSIMs(zData,pars,z2sol,fImpNR)
    % pars
    k1f = pars(1);
    k1b = pars(2);
    k2 = pars(3);
    k3f = pars(4);
    k3b = pars(5);
    k4 = pars(6);
    e0 = pars(7);
    %
    s1 = zData(1,:);
    c1 = zData(2,:);
    s2 = zData(3,:);
    c2 = zData(4,:);

 
    %% Solve SIM expressions for the fast variable (if Explicit)
    if (z2sol(1) == 2) && (z2sol(2) == 4)     % here for c2 and c4
        QSSAc1c2_SIM = zeros(size(z2sol,1),size(zData,2));
        % QSSA
        QSSAc1c2_SIM(1,:) = (e0*k1f*s1*(k4 + k3b))./(k2*k4 + k2*k3b + k4*k1b + k1b*k3b + k4*k1f*s1 + k2*k3f*s2 + k3b*k1f*s1 + k1b*k3f*s2);
        QSSAc1c2_SIM(2,:) = (e0*k3f*s2*(k2 + k1b))./(k2*k4 + k2*k3b + k4*k1b + k1b*k3b + k4*k1f*s1 + k2*k3f*s2 + k3b*k1f*s1 + k1b*k3f*s2);
        % PEA
        PEA13c1c2_SIM = zeros(size(z2sol,1),size(zData,2));
        if ~fImpNR
            PEA13c1c2_SIM(1,:) = (-((c1 + c2 - e0)*k1f.*(c1*k1b + (c1 + c2 - e0)*k1f.*s1).*(-k3b + k3f*(c1 + c2 - e0 - s2))) - (c1 + c2 - e0)*k1f*k3f.*s1.*(c2*k3b + (c1 + c2 - e0)*k3f.*s2) + (c1 + c2 - e0)*k1f*k3f.*s1.*(c2*(k3b + k4) + ...
                (c1 + c2 - e0)*k3f.*s2) + (c1*(k1b + k2) + (c1 + c2 - e0)*k1f.*s1).*(-((k3b - (c1 + c2 - e0)*k3f).*(k1b + k1f*s1)) - k1b*k3f*s2))./((k3b - (c1 + c2 - e0)*k3f).*(k1b + k1f*(-c1 - c2 + e0 + s1)) + (k1b - (c1 + c2 - e0)*k1f)*k3f.*s2);
            PEA13c1c2_SIM(2,:) = (-((c1 + c2 - e0)*k1f*k3f.*(c1*k1b + (c1 + c2 - e0)*k1f.*s1).*s2) + (c1 + c2 - e0)*k1f*k3f.*(c1*(k1b + k2) + (c1 + c2 - e0)*k1f.*s1).*s2 - (c1 + c2 - e0)*k3f.*(-k1b + k1f*(c1 + c2 - e0 - s1)).*(c2*k3b + ...
                (c1 + c2 - e0)*k3f.*s2) + (k3b*(k1b + k1f*(-c1 - c2 + e0 + s1)) + (k1b - (c1 + c2 - e0)*k1f)*k3f.*s2).*((-c1 + e0)*k3f.*s2 - c2.*(k3b + k4 + k3f*s2)))./((k3b - (c1 + c2 - e0)*k3f).*(k1b + k1f*(-c1 - c2 + e0 + s1)) + (k1b - ...
                (c1 + c2 - e0)*k1f)*k3f.*s2); 
        else
            tic
            x_init = [1; 1];                                                                                                            
            for i = 1:size(zData,2)
                PEA13c1c2_SIM(1:2,i) = rootNR(@PEA13c1c2_impC1C2,[s1(1,i) ; s2(1,i)],pars,x_init);
            end
            CPUend = toc;
            fprintf('Solving PEA with NR took %e  \n',CPUend);
        end    
        % CSP
        % CSPc1c2: (1,1) 
        CSP11c1c2_SIM = zeros(size(z2sol,1),size(zData,2));
        if ~fImpNR
            CSP11c1c2_SIM(1,:) = -(c1*k1b) - c1*k2 - (c1 + c2 - e0)*k1f.*s1 + ((c1 + c2 - e0)*k1f.*(c1*k1b + (c1 + c2 - e0)*k1f.*s1).*(k3b + k4 + k3f*s2))./((k3b + k4)*(k1b + k2 + k1f*s1) + (k1b + k2)*k3f*s2) - ((c1 + c2 - e0)*k1f*k3f.*s1.*(c2*k3b + ...
                (c1 + c2 - e0)*k3f.*s2))./((k3b + k4)*(k1b + k2 + k1f*s1) + (k1b + k2)*k3f*s2);
            CSP11c1c2_SIM(2,:) = -(c2*k3b) - c2*k4 - (c1 + c2 - e0)*k3f.*s2 - ((c1 + c2 - e0)*k1f*k3f.*(c1*k1b + (c1 + c2 - e0)*k1f.*s1).*s2)./((k3b + k4)*(k1b + k2 + k1f*s1) + (k1b + k2)*k3f*s2) + ((c1 + c2 - e0)*k3f.*(k1b + k2 + k1f*s1).*(c2*k3b + ...
                (c1 + c2 - e0)*k3f.*s2))./((k3b + k4)*(k1b + k2 + k1f*s1) + (k1b + k2)*k3f*s2);
        else
            x_init = [1; 1];
            tic
            for i = 1:size(zData,2)
                CSP11c1c2_SIM(1:2,i) = rootNR(@CSP11c1c2_impC1C2,[s1(1,i) ; s2(1,i)],pars,x_init);
            end
            CPUend = toc;
            fprintf('Solving CSP11 with NR took %e  \n',CPUend);
        end
        % CSPc1c2: (2,1)
        CSP21c1c2_SIM = zeros(size(z2sol,1),size(zData,2));
        %% Only numerically
        if ~fImpNR
            for i = 1:size(zData,2)
                [Ar11, As11, Br11, Bs11, Ar21, As21, Br21, Bs21, RHS] = CSP1n2_BV([c1(1,i);c2(1,i);s1(1,i);s2(1,i)],pars);
                CSP21c1c2_SIM(:,i) = Br21*RHS;
            end
        else
            x_init = [1; 1];
            tic
            for i = 1:size(zData,2)
                CSP21c1c2_SIM(1:2,i) = rootNR(@CSP21c1c2_impC1C2,[s1(1,i) ; s2(1,i)],pars,x_init);
            end
            CPUend = toc;
            fprintf('Solving CSP21 with NR took %e  \n',CPUend);
        end
    end
    
end

%%% Newton-Raphson iterative scheme %%%  <---------- function functions
function xNR = rootNR(funHandle,ySlow,pars,x0)
    tol = 1e-6;
    dx = 100;
    iter = 0;
    while (norm(dx)>tol) && (iter<10)
        [fun, Jac] = funHandle(x0,ySlow,pars);
        dx = -Jac\fun;
        % Newton Update
        x0 = x0 + dx;
        iter = iter + 1;
        if iter == 10
            error('I want more iterations, Newton said');
        end
    end
    xNR = x0;

end


%%% FUNCTION handlers to use in NR %%%
%
%
%

% PEA13c1c2 solved for c1, c2
function [PEA13c1c2_SIM, Jac_PEA13c1c2_SIM] = PEA13c1c2_impC1C2(xFast,ySlow,pars)
    k1f = pars(1);
    k1b = pars(2);
    k2 = pars(3);
    k3f = pars(4);
    k3b = pars(5);
    k4 = pars(6);
    e0 = pars(7);
    % in this case 
    c1 = xFast(1);
    c2 = xFast(2);
    s1 = ySlow(1);
    s2 = ySlow(2);
    % 
    PEA13c1c2_SIM(1,1) = (-((c1 + c2 - e0)*k1f*(c1*k1b + (c1 + c2 - e0)*k1f*s1)*(-k3b + k3f*(c1 + c2 - e0 - s2))) - (c1 + c2 - e0)*k1f*k3f*s1*(c2*k3b + (c1 + c2 - e0)*k3f*s2) + (c1 + c2 - e0)*k1f*k3f*s1*(c2*(k3b + k4) + ...
        (c1 + c2 - e0)*k3f*s2) + (c1*(k1b + k2) + (c1 + c2 - e0)*k1f*s1)*(-((k3b - (c1 + c2 - e0)*k3f)*(k1b + k1f*s1)) - k1b*k3f*s2))/((k3b - (c1 + c2 - e0)*k3f)*(k1b + k1f*(-c1 - c2 + e0 + s1)) + (k1b - (c1 + c2 - e0)*k1f)*k3f*s2);
    PEA13c1c2_SIM(2,1) = (-((c1 + c2 - e0)*k1f*k3f*(c1*k1b + (c1 + c2 - e0)*k1f*s1)*s2) + (c1 + c2 - e0)*k1f*k3f*(c1*(k1b + k2) + (c1 + c2 - e0)*k1f*s1)*s2 - (c1 + c2 - e0)*k3f*(-k1b + k1f*(c1 + c2 - e0 - s1))*(c2*k3b + ...
        (c1 + c2 - e0)*k3f*s2) + (k3b*(k1b + k1f*(-c1 - c2 + e0 + s1)) + (k1b - (c1 + c2 - e0)*k1f)*k3f*s2)*((-c1 + e0)*k3f*s2 - c2*(k3b + k4 + k3f*s2)))/((k3b - (c1 + c2 - e0)*k3f)*(k1b + k1f*(-c1 - c2 + e0 + s1)) + (k1b - ...
        (c1 + c2 - e0)*k1f)*k3f*s2); 
    %
    % derivatives over c1
    Jac_PEA13c1c2_SIM(1,1) = (((k3b - (c1 + c2 - e0)*k3f)*(k1b + k1f*(-c1 - c2 + e0 + s1)) + (k1b - (c1 + c2 - e0)*k1f)*k3f*s2)*(-((c1 + c2 - e0)*k1f*k3f*(c1*k1b + (c1 + c2 - e0)*k1f*s1)) + k3f*(k1b + k1f*s1)*(c1*(k1b + k2) + ...
        (c1 + c2 - e0)*k1f*s1) - (c1 + c2 - e0)*k1f*(k1b + k1f*s1)*(-k3b + k3f*(c1 + c2 - e0 - s2)) - k1f*(c1*k1b + (c1 + c2 - e0)*k1f*s1)*(-k3b + k3f*(c1 + c2 - e0 - s2)) - k1f*k3f*s1*(c2*k3b + (c1 + c2 - e0)*k3f*s2) + ...
        k1f*k3f*s1*(c2*(k3b + k4) + (c1 + c2 - e0)*k3f*s2) + (k1b + k2 + k1f*s1)*(-((k3b - (c1 + c2 - e0)*k3f)*(k1b + k1f*s1)) - k1b*k3f*s2)) - (-((c1 + c2 - e0)*k1f*(c1*k1b + (c1 + c2 - e0)*k1f*s1)*(-k3b + k3f*(c1 + c2 - ...
        e0 - s2))) - (c1 + c2 - e0)*k1f*k3f*s1*(c2*k3b + (c1 + c2 - e0)*k3f*s2) + (c1 + c2 - e0)*k1f*k3f*s1*(c2*(k3b + k4) + (c1 + c2 - e0)*k3f*s2) + (c1*(k1b + k2) + (c1 + c2 - e0)*k1f*s1)*(-((k3b - (c1 + c2 - ...
        e0)*k3f)*(k1b + k1f*s1)) - k1b*k3f*s2))*(-(k1b*k3f) - k1f*(k3b + k3f*(-2*c1 - 2*c2 + 2*e0 + s1 + s2))))/((k3b - (c1 + c2 - e0)*k3f)*(k1b + k1f*(-c1 - c2 + e0 + s1)) + (k1b - (c1 + c2 - e0)*k1f)*k3f*s2)^2;
    Jac_PEA13c1c2_SIM(2,1) = (-((-(k1b*k3f) - k1f*(k3b + k3f*(-2*c1 - 2*c2 + 2*e0 + s1 + s2)))*(-((c1 + c2 - e0)*k1f*k3f*(c1*k1b + (c1 + c2 - e0)*k1f*s1)*s2) + (c1 + c2 - e0)*k1f*k3f*(c1*(k1b + k2) + (c1 + c2 - ...
        e0)*k1f*s1)*s2 - (c1 + c2 - e0)*k3f*(-k1b + k1f*(c1 + c2 - e0 - s1))*(c2*k3b + (c1 + c2 - e0)*k3f*s2) + (k3b*(k1b + k1f*(-c1 - c2 + e0 + s1)) + (k1b - (c1 + c2 - e0)*k1f)*k3f*s2)*((-c1 + e0)*k3f*s2 - ...
        c2*(k3b + k4 + k3f*s2)))) + ((k3b - (c1 + c2 - e0)*k3f)*(k1b + k1f*(-c1 - c2 + e0 + s1)) + (k1b - (c1 + c2 - e0)*k1f)*k3f*s2)*(-(c2^2*k1f*k3f*(2*k3b + 3*k3f*s2)) + c2*(k1b*k3f*(k3b + 2*k3f*s2) + ...
        k1f*(k3b^2 + k3f*s2*(k2 + k4 + 2*k3f*(-3*c1 + 3*e0 + s1 + s2)) + k3b*(k4 + k3f*(-2*c1 + 2*e0 + s1 + 3*s2)))) + k3f*s2*(-3*c1^2*k1f*k3f - 3*e0^2*k1f*k3f - k3b*(k1b + k1f*s1) - k1b*k3f*s2 + ...
        2*c1*(k1b*k3f + k1f*(k2 + k3b + k3f*(3*e0 + s1 + s2))) - e0*(2*k1b*k3f + k1f*(k2 + 2*(k3b + k3f*(s1 + s2)))))))/((k3b - (c1 + c2 - e0)*k3f)*(k1b + k1f*(-c1 - c2 + e0 + s1)) + (k1b - (c1 + c2 - e0)*k1f)*k3f*s2)^2; 
    %
    Jac_PEA13c1c2_SIM(1,2) = (-((-((c1 + c2 - e0)*k1f*(c1*k1b + (c1 + c2 - e0)*k1f*s1)*(-k3b + k3f*(c1 + c2 - e0 - s2))) - (c1 + c2 - e0)*k1f*k3f*s1*(c2*k3b + (c1 + c2 - e0)*k3f*s2) + (c1 + c2 - ...
        e0)*k1f*k3f*s1*(c2*(k3b + k4) + (c1 + c2 - e0)*k3f*s2) + (c1*(k1b + k2) + (c1 + c2 - e0)*k1f*s1)*(-((k3b - (c1 + c2 - e0)*k3f)*(k1b + k1f*s1)) - k1b*k3f*s2))*(-(k1b*k3f) - ...
        k1f*(k3b + k3f*(-2*c1 - 2*c2 + 2*e0 + s1 + s2)))) + ((k3b - (c1 + c2 - e0)*k3f)*(k1b + k1f*(-c1 - c2 + e0 + s1)) + (k1b - (c1 + c2 - e0)*k1f)*k3f*s2)*(-(c1^2*k1f*k3f*(2*k1b + 3*k1f*s1)) - ...
        k1f*s1*(2*e0*k1f*k3b + 3*c2^2*k1f*k3f + 3*e0^2*k1f*k3f + e0*k3f*k4 + k1f*k3b*s1 + 2*e0*k1f*k3f*s1 + 2*e0*k1f*k3f*s2 + k1b*(k3b + k3f*(-2*c2 + 2*e0 + s2)) - 2*c2*(k3f*k4 + k1f*(k3b + ...
        k3f*(3*e0 + s1 + s2)))) + c1*(k1b^2*k3f + k1f*s1*(k3f*(k2 + k4) + 2*k1f*(k3b + k3f*(-3*c2 + 3*e0 + s1 + s2))) + k1b*(k2*k3f + k1f*(k3b + k3f*(-2*c2 + 2*e0 + 3*s1 + s2))))))/((k3b - ...
        (c1 + c2 - e0)*k3f)*(k1b + k1f*(-c1 - c2 + e0 + s1)) + (k1b - (c1 + c2 - e0)*k1f)*k3f*s2)^2;
    Jac_PEA13c1c2_SIM(2,2) = (((k3b - (c1 + c2 - e0)*k3f)*(k1b + k1f*(-c1 - c2 + e0 + s1)) + (k1b - (c1 + c2 - e0)*k1f)*k3f*s2)*(-(k1f*k3f*(c1*k1b + (c1 + c2 - e0)*k1f*s1)*s2) + k1f*k3f*(c1*(k1b + k2) + ...
        (c1 + c2 - e0)*k1f*s1)*s2 - (c1 + c2 - e0)*k3f*(-k1b + k1f*(c1 + c2 - e0 - s1))*(k3b + k3f*s2) - (c1 + c2 - e0)*k1f*k3f*(c2*k3b + (c1 + c2 - e0)*k3f*s2) - k3f*(-k1b + k1f*(c1 + c2 - e0 - ...
        s1))*(c2*k3b + (c1 + c2 - e0)*k3f*s2) + k1f*(k3b + k3f*s2)*(c2*(k3b + k4) + (c1 + c2 - e0)*k3f*s2) + (-k3b - k4 - k3f*s2)*(k3b*(k1b + k1f*(-c1 - c2 + e0 + s1)) + (k1b - (c1 + c2 - e0)*k1f)*k3f*s2)) - ...
        (-(k1b*k3f) - k1f*(k3b + k3f*(-2*c1 - 2*c2 + 2*e0 + s1 + s2)))*(-((c1 + c2 - e0)*k1f*k3f*(c1*k1b + (c1 + c2 - e0)*k1f*s1)*s2) + (c1 + c2 - e0)*k1f*k3f*(c1*(k1b + k2) + (c1 + c2 - e0)*k1f*s1)*s2 - ...
        (c1 + c2 - e0)*k3f*(-k1b + k1f*(c1 + c2 - e0 - s1))*(c2*k3b + (c1 + c2 - e0)*k3f*s2) + (k3b*(k1b + k1f*(-c1 - c2 + e0 + s1)) + (k1b - (c1 + c2 - e0)*k1f)*k3f*s2)*((-c1 + e0)*k3f*s2 - ...
        c2*(k3b + k4 + k3f*s2))))/((k3b - (c1 + c2 - e0)*k3f)*(k1b + k1f*(-c1 - c2 + e0 + s1)) + (k1b - (c1 + c2 - e0)*k1f)*k3f*s2)^2 ;
    
end

% CSP11c1c2 solved for c1, c2
function [CSP11c1c2_SIM, Jac_CSP11c1c2_SIM] = CSP11c1c2_impC1C2(xFast,ySlow,pars)
    k1f = pars(1);
    k1b = pars(2);
    k2 = pars(3);
    k3f = pars(4);
    k3b = pars(5);
    k4 = pars(6);
    e0 = pars(7);
    % in this case 
    c1 = xFast(1);
    c2 = xFast(2);
    s1 = ySlow(1);
    s2 = ySlow(2);
    % 
    CSP11c1c2_SIM(1,1) = -(c1*k1b) - c1*k2 - (c1 + c2 - e0)*k1f*s1 + ((c1 + c2 - e0)*k1f*(c1*k1b + (c1 + c2 - e0)*k1f*s1)*(k3b + k4 + k3f*s2))/((k3b + k4)*(k1b + k2 + k1f*s1) + (k1b + k2)*k3f*s2) - ((c1 + c2 - e0)*k1f*k3f*s1*(c2*k3b + ...
        (c1 + c2 - e0)*k3f*s2))/((k3b + k4)*(k1b + k2 + k1f*s1) + (k1b + k2)*k3f*s2);
    CSP11c1c2_SIM(2,1) = -(c2*k3b) - c2*k4 - (c1 + c2 - e0)*k3f*s2 - ((c1 + c2 - e0)*k1f*k3f*(c1*k1b + (c1 + c2 - e0)*k1f*s1)*s2)/((k3b + k4)*(k1b + k2 + k1f*s1) + (k1b + k2)*k3f*s2) + ((c1 + c2 - e0)*k3f*(k1b + k2 + k1f*s1)*(c2*k3b + ...
        (c1 + c2 - e0)*k3f*s2))/((k3b + k4)*(k1b + k2 + k1f*s1) + (k1b + k2)*k3f*s2);  
            %
    % derivatives over c1
    Jac_CSP11c1c2_SIM(1,1) = -(((c1 + c2 - e0)*k1f*k3f^2*s1*s2 - (c1 + c2 - e0)*k1f*(k1b + k1f*s1)*(k3b + k4 + k3f*s2) - k1f*(c1*k1b + (c1 + c2 - e0)*k1f*s1)*(k3b + k4 + k3f*s2) + k1f*k3f*s1*(c2*k3b + (c1 + c2 - e0)*k3f*s2) + ...
        k1b*((k3b + k4)*(k1b + k2 + k1f*s1) + (k1b + k2)*k3f*s2) + k2*((k3b + k4)*(k1b + k2 + k1f*s1) + (k1b + k2)*k3f*s2) + k1f*s1*((k3b + k4)*(k1b + k2 + k1f*s1) + (k1b + k2)*k3f*s2))/((k3b + k4)*(k1b + k2 + k1f*s1) + ...
        (k1b + k2)*k3f*s2));
    Jac_CSP11c1c2_SIM(2,1) = (k3f*(c2*k3b*(k1b + k2 + k1f*s1) - (2*c1*k1b*k1f + c2*k1b*k1f - e0*k1b*k1f + k1b*k3b + k2*k3b - 2*c1*k1b*k3f - 2*c2*k1b*k3f + 2*e0*k1b*k3f - 2*c1*k2*k3f - 2*c2*k2*k3f + 2*e0*k2*k3f + k1b*k4 + k2*k4 + ...
        k1f*(k3b + 2*(c1 + c2 - e0)*(k1f - k3f) + k4)*s1)*s2 - (k1b + k2)*k3f*s2^2))/((k3b + k4)*(k1b + k2 + k1f*s1) + (k1b + k2)*k3f*s2);
    %
    Jac_CSP11c1c2_SIM(1,2) = (k1f*(-(s1*(2*e0*k1f*k3b + k2*k3b + 2*c2*k3b*k3f - e0*k3b*k3f + 2*e0*k1f*k4 + k2*k4 + k1b*(k3b + k4) - 2*c2*k1f*(k3b + k4) + k1f*(k3b + k4)*s1)) - (k1b + k2 - 2*(c2 - e0)*(k1f - k3f))*k3f*s1*s2 + ...
        c1*(k1b*(k3b + k4 + k3f*s2) + 2*k1f*s1*(k3b + k4 + k3f*s2) - k3f*s1*(k3b + 2*k3f*s2))))/((k3b + k4)*(k1b + k2 + k1f*s1) + (k1b + k2)*k3f*s2);
    Jac_CSP11c1c2_SIM(2,2) = -(((k3b*(k3b + (-c1 - 2*c2 + e0)*k3f) + 2*k3b*k4 + k4^2)*(k2 + k1f*s1) + k3f*(2*k2*(k3b - (c1 + c2 - e0)*k3f + k4) + k1f*(k3b + 2*(c1 + c2 - e0)*(k1f - k3f) + k4)*s1)*s2 + k2*k3f^2*s2^2 + ...
        k1b*(k3b^2 + k3f*(c1*k1f - 2*(c1 + c2 - e0)*k3f)*s2 + (k4 + k3f*s2)^2 + k3b*(2*k4 + k3f*(-c1 - 2*c2 + e0 + 2*s2))))/((k3b + k4)*(k1b + k2 + k1f*s1) + (k1b + k2)*k3f*s2));
end

% CSP21c1c2 solved for c1, c2
function [CSP21c1c2_SIM, Jac_CSP21c1c2_SIM] = CSP21c1c2_impC1C2(xFast,ySlow,pars)
    k1f = pars(1);
    k1b = pars(2);
    k2 = pars(3);
    k3f = pars(4);
    k3b = pars(5);
    k4 = pars(6);
    e0 = pars(7);
    % in this case 
    c1 = xFast(1);
    c2 = xFast(2);
    s1 = ySlow(1);
    s2 = ySlow(2);
    % 
    %% For f do the refinements
    [~, ~, ~, ~, ~, ~, Br21, ~, RHS] = CSP1n2_BV([c1; c2 ;s1 ;s2],pars);
    CSP21c1c2_SIM = Br21*RHS;
    %
    %% For the jacobian do central finite differences!
    pertPC = 1e-3;
    % derivative on c1
    c1PertP = c1*(1+pertPC);
    [~, ~, ~, ~, ~, ~, Br21, ~, RHS] = CSP1n2_BV([c1PertP; c2 ;s1 ;s2],pars);
    funP1 = Br21*RHS;
    c1PertM = c1*(1-pertPC);
    [~, ~, ~, ~, ~, ~, Br21, ~, RHS] = CSP1n2_BV([c1PertM; c2 ;s1 ;s2],pars);
    funM1 = Br21*RHS;
    Jac_CSP21c1c2_SIM(:,1) = (funP1-funM1)/(2*c1*pertPC);
    % derivative on c2
    c2PertP = c2*(1+pertPC);
    [~, ~, ~, ~, ~, ~, Br21, ~, RHS] = CSP1n2_BV([c1; c2PertP ;s1 ;s2],pars);
    funP2 = Br21*RHS;
    c2PertM = c2*(1-pertPC);
    [~, ~, ~, ~, ~, ~, Br21, ~, RHS] = CSP1n2_BV([c1; c2PertM ;s1 ;s2],pars);
    funM2 = Br21*RHS;
    Jac_CSP21c1c2_SIM(:,2) = (funP2-funM2)/(2*c2*pertPC);

end

%%%%%%%%%%%%%%%%%%%%%%%% FUNCTIONS TO CALCULATE CSP21c1c2 numerically

% get the basis vectors after the 1st-stage and after the 2nd stage
function [Ar11, As11, Br11, Bs11, Ar21, As21, Br21, Bs21, RHS] = CSP1n2_BV(zSol,pars)
    c1 = zSol(1);
    c2 = zSol(2);
    s1 = zSol(3);
    s2 = zSol(4);
    Ntot = size(zSol,1);
    %
    k1f = pars(1);
    k1b = pars(2);
    k2 = pars(3);
    k3f = pars(4);
    k3b = pars(5);
    k4 = pars(6);
    e0 = pars(7);
    %
    Ar00 = [1 0; 0 1; 0 0; 0 0];
    As00 = [0 0; 0 0; 1 0; 0 1];
    Br00 = [1 0 0 0; 0 1 0 0];
    Bs00 = [0 0 1 0; 0 0 0 1];
    %
    RHS = INHode(0.,[s1; c1; s2; c2],k1f,k1b,k2,k3f,k3b,k4,e0);
    Jac = gradINHode(0.,[s1; c1; s2; c2],k1f,k1b,k2,k3f,k3b,k4,e0);
    T1 = [0 1 0 0; 0 0 0 1; 1 0 0 0; 0 0 1 0];
    RHS = T1*RHS;       % c1 , c2, s1, s2
    Jac = T1*Jac*T1';   % c1 , c2, s1, s2
    dJacdt = [0 0 -k1f 0; 0 0 0 -k3f; 0 0 k1f 0; 0 0 0 k3f]*(RHS(1,1) + RHS(2,1)) + [-k1f -k1f 0 0; 0 0 0 0; k1f k1f 0 0; 0 0 0 0]*RHS(3,1) + [0 0 0 0; -k3f -k3f 0 0; 0 0 0 0; k3f k3f 0 0]*RHS(4,1); 
    %
    %% 1st phase
    % Br-ref
    Lambda00 = Br00*Jac*Ar00;
    Tau00 = inv(Lambda00);
    Br10 = Tau00*Br00*Jac;
    Ar10 = Ar00;
    Bs10 = Bs00;
    As10 = (eye(Ntot) - Ar10*Br10)*As00;
    % Ar-ref
    Lambda10 = Br10*Jac*Ar10;
    Tau10 = inv(Lambda10);
    Ar11 = Jac*Ar10*Tau10;
    Br11 = Br10;
    As11 = As10;
    Bs11 = Bs10*(eye(Ntot) - Ar11*Br11);
    %
    %% 2nd phase
    dBr11dt = Tau00*Br00*dJacdt*(eye(Ntot)-Ar10*Br10);
    Lambda11 = (dBr11dt+Br11*Jac)*Ar11;
    Tau11 = inv(Lambda11);
    Br21 = Tau11*(dBr11dt+Br11*Jac);
    Ar21 = Ar11;
    Bs21 = Bs11;
    As21 = (eye(Ntot) - Ar21*Br21)*As11;           

end
