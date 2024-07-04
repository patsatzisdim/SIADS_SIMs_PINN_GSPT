%% fCSI system of ODEs, transformed variables
function dydt = INHode(t,y,k1f,k1b,k2,k3f,k3b,k4,e0)
    nVar = size(y,1);
    nPoints = size(y,2);
    dydt = zeros(nVar,nPoints);
    dydt(1,:) = - k2*y(2,:);                                                         % ds1BARdt
    dydt(2,:) =  k1f*(e0-y(2,:)-y(4,:)).*(y(1,:)-y(2,:)) - k1b*y(2,:) - k2*y(2,:);   % dc1dt
    dydt(3,:) = - k4*y(4,:);                                                         % ds2BARdt
    dydt(4,:) =  k3f*(e0-y(2,:)-y(4,:)).*(y(3,:)-y(4,:)) - k3b*y(4,:) - k4*y(4,:);   % dc2dt
    
end

