%% Jacobia of the MM system of ODEs
function [Jac_x, Jac_y] = gradMMode(t,y,k1f,k1b,k2,e0)
    nVar = size(y,1);
    nPoints = size(y,2);
    Jac_x = zeros(nVar,nPoints);
    Jac_y = zeros(nVar,nPoints);

    Jac_x(1,:) = -k1f*y(2,:)-k1b-k2;            % grad dcdt over c 
    Jac_y(1,:) = k1f*(e0-y(1,:));               % grad dcdt over s
    
    Jac_x(2,:) = k1f*y(2,:)+k1b;                % grad dsdt over c
    Jac_y(2,:) = -k1f*(e0-y(1,:));              % grad dsdt over s

end