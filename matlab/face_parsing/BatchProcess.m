function batch = BatchProcess(batch)
batch = {single(batch)};
%batch = {single(permute(batch,[2 1 3 4]))};    % flip width and height

end
