function [x] = conjgrad(A, b)
    %Initialize with zeros
    x = zeros(size(A,2),1);
    
    %TO DO: In this line you have to change the product of matrix A and
    %vector x with the trick in Andrew Ng's paper. Do two functions: one to
    %compute the trick for the gradient of the training cost with respect
    %to the weights, and another with the gradient of the training cost
    %with respect to the poisoning point. Then, you have to change the
    %argument of the function, i.e. you don't need A anymore, but the
    %aforementioned gradients.
    r = b - A * x;
    
    p = r;
    rsold = r' * r;

    for i = 1:length(b)
        %TODO: Here you have the apply the same trick again
        Ap = A * p;
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