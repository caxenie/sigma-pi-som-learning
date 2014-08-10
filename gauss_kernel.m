% computes the Gaussian kernel given params
function y = compute_kernel(Norm, n, s)
    y = zeros(n, 1);
    for idx = 1:n
       y(idx) = Norm * exp(-(idx-n)^2/(2*s^2)); 
    end
end