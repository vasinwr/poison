function [w, cost] = trainAdaline(x,y,rate,iter)
%Train a Logistic Regression classifier
%output w - column vector

f = size(x,2);
D = size(x,1);

w = zeros(f+1,1);

%Include the bias term in the training data
x = [ones(D,1) x];

%Monitor the cost with the number of iterations
cost = zeros(iter,1);

for i=1:iter
    dw = mean(repmat(x*w-y,1,f+1).*x)'; %dw column vector
    w = w - rate*dw;    
    cost(i) = mean((x*w - y).^2);
end

