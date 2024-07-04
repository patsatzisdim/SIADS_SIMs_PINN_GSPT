%% Jacobian of the TMDD mechanism
function Jac = gradTMDDode(t,y,kon,koff,kel,ksyn,kdeg,kint)
    nVar = size(y,1);
    nPoints = size(y,2);
    Jac = zeros(nVar,nVar,nPoints);

    % grad dLdt over L, R, RL
    Jac(1,1,:) = -kon*y(2,:)-kel;             
    Jac(1,2,:) = -kon*y(1,:);                
    Jac(1,3,:) = koff;                
    % grad dRdt over L, R, RL
    Jac(2,1,:) = -kon*y(2,:);             
    Jac(2,2,:) = -kon*y(1,:)-kdeg;                
    Jac(2,3,:) = koff;
    % grad dRLdt over L, R, RL
    Jac(3,1,:) = kon*y(2,:);             
    Jac(3,2,:) = kon*y(1,:);                
    Jac(3,3,:) = -koff-kint;

end