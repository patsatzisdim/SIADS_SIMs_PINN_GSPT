%% fCSI system of ODEs, original state variables
function dydt = INHode(t,y,k1f,k1b,k2,k3f,k3b,k4,e0)
    nVar = size(y,1);
    nPoints = size(y,2);
    dydt = zeros(nVar,nPoints);
    dydt(1,:) = -k1f*(e0-y(2,:)-y(4,:)).*y(1,:) + k1b*y(2,:);               % ds1dt
    dydt(2,:) =  k1f*(e0-y(2,:)-y(4,:)).*y(1,:) - k1b*y(2,:) - k2*y(2,:);   % dc1dt
    dydt(3,:) = -k3f*(e0-y(2,:)-y(4,:)).*y(3,:) + k3b*y(4,:);               % ds2dt
    dydt(4,:) =  k3f*(e0-y(2,:)-y(4,:)).*y(3,:) - k3b*y(4,:) - k4*y(4,:);   % dc2dt
    
end

