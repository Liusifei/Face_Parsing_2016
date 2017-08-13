function mask = edge_balance(eb)
[r,c,ch,bz] = size(eb);
mask = zeros(r,c,ch,bz);
for m = 1:bz
  for n = 1:ch % y & x or classes
      edge = eb(:,:,n,m);
      tmp = zeros(r,c);
      tmp(edge<0.1)=1;
      l = length(find(edge<0.1));
      ne = find(edge>=0.1);
      rnl = randperm(length(ne));
      idx = min(4 * l, length(rnl));
      nidx = ne(rnl(1:idx));
      tmp(nidx) = 1;
      mask(:,:,n,m) = tmp;
  end
end

end
