%% Jacobian of the fCSI system of ODEs, transformed variables
function Jac = gradINHode(t,y,k1f,k1b,k2,k3f,k3b,k4,e0)
    nVar = size(y,1);
    nPoints = size(y,2);
    Jac = zeros(nVar,nVar,nPoints);

    % grad ds1BARdt over s1BAR, c1, s2BAR, c2 
    Jac(1,1,:) = 0.;             
    Jac(1,2,:) = -k2;           
    Jac(1,3,:) = 0.;
    Jac(1,4,:) = 0.;  
    % grad dc1dt over s1BAR, c1, s2BAR, c2
    Jac(2,1,:) = k1f*(e0-y(2,:)-y(4,:));             
    Jac(2,2,:) = -k1b + 2*y(2,:)*k1f + y(4,:)*k1f - k2 - k1f*(e0 + y(1,:));              
    Jac(2,3,:) = 0.; 
    Jac(2,4,:) = k1f*(y(2,:) - y(1,:));
    % grad ds2BARdt over s1BAR, c1, s2BAR, c2
    Jac(3,1,:) = 0.;            
    Jac(3,2,:) = 0.;                
    Jac(3,3,:) = 0.;
    Jac(3,4,:) = -k4; 
    % grad dc2dt over s1BAR, c1, s2BAR, c2
    Jac(4,1,:) = 0.;
    Jac(4,2,:) = k3f*(y(4,:) - y(3,:));                
    Jac(4,3,:) = -((y(2,:) + y(4,:) - e0)*k3f);
    Jac(4,4,:) = -k3b + y(2,:)*k3f + 2*y(4,:)*k3f - k4 - k3f*(e0 + y(3,:));
end