%% MM ode system
function dydt = MMode(t,y,k1f,k1b,k2,e0)
    nVar = size(y,1);
    nPoints = size(y,2);
    dydt = zeros(nVar,nPoints);
    dydt(1,:) = k1f*(e0-y(1,:)).*y(2,:)-(k1b+k2)*y(1,:);         % dcdt
    dydt(2,:) = -k1f*(e0-y(1,:)).*y(2,:)+k1b*y(1,:);             % dsdt
end

