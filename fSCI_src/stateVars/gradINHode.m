%% Jacobian of the fCSI system of ODEs, original state variables
function Jac = gradINHode(t,y,k1f,k1b,k2,k3f,k3b,k4,e0)
    nVar = size(y,1);
    nPoints = size(y,2);
    Jac = zeros(nVar,nVar,nPoints);

    % grad ds1dt over s1, c1, s2, c2
    Jac(1,1,:) = -k1f*(e0-y(2,:)-y(4,:));             
    Jac(1,2,:) = k1f*y(1,:) + k1b;           
    Jac(1,3,:) = 0.;
    Jac(1,4,:) = k1f*y(1,:);  
    % grad dc1dt over s1, c1, s2, c2
    Jac(2,1,:) = k1f*(e0-y(2,:)-y(4,:));             
    Jac(2,2,:) = -k1f*y(1,:) - k1b - k2;              
    Jac(2,3,:) = 0.; 
    Jac(2,4,:) = -k1f*y(1,:);
    % grad ds2dt over s1, c1, s2, c2
    Jac(3,1,:) = 0.;            
    Jac(3,2,:) = k3f*y(3,:);                
    Jac(3,3,:) = -k3f*(e0-y(2,:)-y(4,:));
    Jac(3,4,:) = k3f*y(3,:) + k3b; 
    % grad dc2dt over s1, c1, s2, c2
    Jac(4,1,:) = 0.;
    Jac(4,2,:) = -k3f*y(3,:);                
    Jac(4,3,:) = k3f*(e0-y(2,:)-y(4,:));
    Jac(4,4,:) = -k3f*y(3,:) - k3b - k4;
end