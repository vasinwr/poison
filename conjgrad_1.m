function [x] = conjgrad_1(w,Xp,Yp,Xval,Yval)
f = size(Xp,2);
D = size(Xp,1);
b = mean(repmat(Xval*w - Yval,1,f).*Xval)'; % b = dCval/dw
                                               

%Initialize with zeros
    x = zeros(f,1);
    
    %TO DO: In this line you have to change the product of matrix A and
    %vector x with the trick in Andrew Ng's paper. Do two functions: one to
    %compute the trick for the gradient of the training cost with respect
    %to the weights, and another with the gradient of the training cost
    %with respect to the poisoning point. Then, you have to change the
    %argument of the function, i.e. you don't need A anymore, but the
    %aforementioned gradients.
    Aeval = finiteDif(Xp,Yp,w,x);
    r = b - Aeval; % Aeval = A * x = d/dw(d/dw(Ctr)) * x ==> return x when d/dw(d/dw(Ctr)) * x is approximately equals to b (dCval/dw)
    
    p = r;
    rsold = r' * r;

    for i = 1:length(b)
        %TODO: Here you have the apply the same trick again
        Ap = finiteDif(Xp,Yp,w,p); % Ap = A * p
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
% outputs: d/dw(d/dw(Ctr)) * x ; column vector fx1
r = 1e-8;
f = size(Xp,2);
w2 = w + r.*x;
C2 = mean(repmat(Xp*w2 - Yp,1,f).*Xp);
C1 = mean(repmat(Xp*w - Yp,1,f).*Xp);
out = (C2 - C1)./r;
out = out';
end