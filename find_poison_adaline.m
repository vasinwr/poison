function [ x_p ] = find_poison_adaline( X, Y, y_p )
%FIND_POISON Summary of this function goes here
%   Detailed explanation goes here

    %initialise x_p as zero vector
    x_p = zeros(1, size(X,1));
    
    learning_rate = 0.01
    iter = 1500
    
    %while not_minima, for now set as 'iter'
    for i=1:iter
        
        %adding x_p to the training dataset
        Xp = [X;x_p];
        Yp = [Y;y_p];

        w = trainAdaline(Xp,Yp,learning_rate,iter);
        
        %pluck w, Xp, Yp, x_val, y_val,x_p, y_p into the equation
        dCval_dxp = derivative_dCv_dxp(w, Xp, Yp, x_val, y_val, x_p, y_p);
        
        x_p = x_p - x_p*dCval_dxp;
    
    end

    % when loop exited, optimal x_p is found

end

function [dxp] = derivative_dCv_dxp(w, Xp, Yp, x_val, y_val, x_p, y_p)


end
