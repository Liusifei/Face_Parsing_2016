function mask = cls_balance(lb, rate)
% equally sample
dsize = size(lb);
mask = zeros(dsize);
N = dsize(1)*dsize(2);
sele = floor(N * rate / dsize(3));
for m = 1:dsize(4)
    [~,Y] = max(lb(:,:,:,m),[],3);
    tmp = zeros(dsize(1),dsize(2));
    for k = 1:dsize(3)
        l = find(Y==k);
        if length(l) < sele
           tmp(l) = 1; % select all
        else
           idx = randperm(length(l));
           sl = l(idx(1:sele));
           tmp(sl) = 1;
        end
    end
    mask(:,:,:,m) = repmat(tmp,[1,1,dsize(3)]);
end
end
