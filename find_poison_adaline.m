function [ x_p ] = find_poison_adaline( X, Y, Xval, Yval, y_p )
%FIND_POISON Summary of this function goes here
%   Detailed explanation goes here

    %initialise x_p as zero vector
    x_p = zeros(1, size(X,2));
    
    learning_rate = 0.01;
    iter = 1500;
    
    %while not_minima, for now set as 'iter'
    for i=1:iter
        
        %adding x_p poisoning point to the training dataset
        Xp = [X;x_p];
        Yp = [Y;y_p];

        w = trainAdaline(Xp,Yp,learning_rate,iter);
        
        dCval_dxp = find_dCval_dxp(w, Xp, Yp, Xval, Yval, x_p, y_p);
        
        x_p = x_p + learning_rate.*dCval_dxp;
    
    end

    % when loop exited, optimal x_p is found

end

function [dCval_dxp] = find_dCval_dxp(w, Xp, Yp, Xval, Yval, x_p, y_p)
    %input:
    %   w  - weight; column vector
    %   Xp - training dataset with poisoning point in it; 
    %           matrix(row:data point; column:features)
    %   Yp - expected output for Xp; column vector
    %   Xval - validation dataset;
    %           matrix(row:data point; column:features)
    %   Yval - expected output for Yval; column vector
    %   x_p - poisoning point; row vector
    %   y_p - expected output for x_p; scala
    %output:
    %dCval/dxp - the derivative; row vector
    

    n = size(Xp,1);
    
    x = conjgrad_1(w,Xp,Yp,Xval,Yval); % x = (d/dw(d/dw(Ctr))^-1 * d/dw(Cval)
    
    r = 1e-8;
    w2 = w + r.*x;
    c2 = ((w2'*x_p'-y_p).*w2')/n; % c2 = d/dxp(Ctr(w+rx))
    c1 = ((w'*x_p'-y_p).*w')/n;   % c1 = d/dxp(Ctr(w))
    dCval_dxp = -(c2 - c1)./r;    % dCval_dxp = - (d/dxp(d/dw(Ctr))*x)
                                        % where d/dxp(d/dw(Ctr) =
                                        % (c2-c1)./r
                                        % and x = (d/dw(d/dw(Ctr))^-1 * d/dw(Cval)
    
end
