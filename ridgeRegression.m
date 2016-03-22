function w = ridgeRegression( y, X, lambda )
% ridge regression solver
d = size(X,2);
w = inv(X'*X+lambda*eye(d))*X'*y;
end

