function mu = T1_softmax(eta)
    % Softmax function
    % mu(i,c) = exp(eta(i,c))/sum_c' exp(eta(i,c'))
    [h,w,cn,n] = size(eta);
    % This file is from matlabtools.googlecode.com
    mu = zeros(h,w,cn,n);
    for ind = 1:n
        aa = eta(:,:,:,ind);
        aa = reshape(aa,[h*w,cn]);
        c = 1;
%         eta = eta';
        tmp = exp(c*aa);
        denom = sum(tmp, 2);
        m = bsxfun(@rdivide, tmp, denom);
%         m = m';
        mu(:,:,:,ind) = reshape(m,[h,w,cn]);
    end
end