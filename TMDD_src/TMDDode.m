%% TMDD system of odes
function dydt = TMDDode(t,y,kon,koff,kel,ksyn,kdeg,kint)
    nVar = size(y,1);
    nPoints = size(y,2);
    dydt = zeros(nVar,nPoints);
    dydt(1,:) = -kon*y(1,:).*y(2,:)+koff*y(3,:)-kel*y(1,:);                 % dLdt
    dydt(2,:) = -kon*y(1,:).*y(2,:)+koff*y(3,:)+ksyn-kdeg*y(2,:);           % dRdt
    dydt(3,:) = kon*y(1,:).*y(2,:)-koff*y(3,:)-kint*y(3,:);                 % dRLdt
end

