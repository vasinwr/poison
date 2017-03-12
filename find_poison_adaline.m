function [ x_p ] = find_poison_adaline( X, Y, y_p )
%FIND_POISON Summary of this function goes here
%   Detailed explanation goes here

    %initialise x_p as zero vector
    x_p = zeros(1, size(X,2));
    
    learning_rate = 0.01;
    iter = 1500;
    
    %while not_minima, for now set as 'iter'
    for i=1:iter
        
        %adding x_p to the training dataset
        Xp = [X;x_p];
        Yp = [Y;y_p];

        w = trainAdaline(Xp,Yp,learning_rate,iter);
        
        %pluck w, Xp, Yp, x_val, y_val,x_p, y_p into the equation
        dCval_dxp = find_dCval_dxp(w, Xp, Yp, x_val, y_val, x_p, y_p);
        
        x_p = x_p - x_p.*dCval_dxp;
    
    end

    % when loop exited, optimal x_p is found

end

function [dCval_dxp] = find_dCval_dxp(w, Xp, Yp, x_val, y_val, x_p, y_p)
    %input:
    %   w  - weight; column vector
    %   Xp - training dataset with poisoning point in it; 
    %           matrix(row:data point; column:features)
    %   Yp - expected output for Xp; column vector
    %   x_val - validation dataset;
    %           matrix(row:data point; column:features)
    %   y_val - expected output for y_val; column vector
    %   x_p - poisoning point; row vector
    %   y_p - expected output for x_p; scala
    %output:
    %dCval/dxp - the derivative; row vector
    

    n = size(Xp,2);
    
    %dCval/dxp = dCval/dw * dw/dxp
    % (col vector) = (row vector) * (matrix)
    
    %dCval/dw; column vector
    dCval_dw = mean((w'*x_val'-y_val').*x_val',2);

    %dw/dxp = dw/dCtr * dCtr/dxp
    %dCtr/dw; column vector
    dCtr_dw = mean((w'*Xp'-Yp').*Xp',2);
    dw_dCtr = 1./dCtr_dw % is this okay?
    
    %dCtr/dxp; column vector
    dCtr_dxp = ((w'*x_p'-y_p).*w')/n;
    
    %dw/dxp; (matrix) = (column vec) * (row vec)
    dw_dxp = dw_dCtr' * dCtr_dxp;
    
    %row vector
    dCval_dxp = dCval_dw' * dw_dxp;
end
