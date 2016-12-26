function data_ = get_data(Solver_)
% add by Sifei Liu
net_ = Solver_.net;
attr = net_.get_attr();
num = length(attr.hBlob_blobs);
for m = 1:num
   data_(m).name = attr.blob_names(attr.input_blob_indices + m);
   blob = net_.blobs(data_(m).name{1});
   data_(m).data = blob.get_data();
   data_(m).diff = blob.get_diff();
end
end