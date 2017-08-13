function mask = uni_balance(dsize, rate)
mask = zeros(dsize);
for m = 1:dsize(4)
    msk = rand(dsize(1),dsize(2)) < rate;
    mask(:,:,:,m) = repmat(msk,[1,1,dsize(3)]);
end
end