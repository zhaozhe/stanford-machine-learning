function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


htheta = sigmoid(X*theta);
part1 = y.* log(htheta);
part2 = (1-y).*log(1- htheta);

J = -1* sum (part1 + part2) / m;

part3 = 0;
for i = 2:size(theta)
	part3 = part3 + theta(i)^2;
end
part3 = part3 * lambda / (2*m);

J = J + part3;

grad = (X' * (htheta-y))*(1/m);

for j = 2:size(theta)
	grad(j) = grad(j) + lambda/m*theta(j);
end

% =============================================================

end
