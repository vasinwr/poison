function [x] = conjgrad_1(w,Xp,Yp,x_val,y_val)
D = size(Xp,2);
b = mean(repmat(w'*x_val' - y_val,1,D).*x_val);    

%Initialize with zeros
    x = zeros(D,1);
    
    %TO DO: In this line you have to change the product of matrix A and
    %vector x with the trick in Andrew Ng's paper. Do two functions: one to
    %compute the trick for the gradient of the training cost with respect
    %to the weights, and another with the gradient of the training cost
    %with respect to the poisoning point. Then, you have to change the
    %argument of the function, i.e. you don't need A anymore, but the
    %aforementioned gradients.
    Aeval = finiteDif(Xp,Yp,w,x);
    r = b - Aeval;
    
    p = r;
    rsold = r' * r;

    for i = 1:length(b)
        %TODO: Here you have the apply the same trick again
        Ap = finiteDif(Xp,Yp,w,p);
        alpha = rsold / (p' * Ap);
        x = x + alpha * p;
        r = r - alpha * Ap;
        rsnew = r' * r;
        if sqrt(rsnew) < 1e-10
              break;
        end
        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    end
end

function [out] = finiteDif(Xp,Yp,w,x)
r = 1e-8;
D = size(Xp,2);
w2 = w + r.*x;
C2 = mean(repmat(w2'*Xp - Yp,1,D).*Xp);
C1 = mean(repmat(w'*Xp - Yp,1,D).*Xp);
out = (C2 - C1)./r;
end